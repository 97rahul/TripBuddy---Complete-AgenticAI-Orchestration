import asyncio
from typing import TypedDict, Annotated, Any
import operator
import uuid
import os
# from tools.flight_tool import search_flights
# from tools.tavily_tool import tavily_search
from mcp_tools import tavily_mcp_search,aviation_mcp_call, weather_mcp_call
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,AnyMessage
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from utils import (
    AGENT_ORDER,
    KNOWN_AGENTS,
    empty_constraints,
    extract_destination,
    json_from_llm,
    llm_text,
    render_prompt,
    route_after_agent,
    route_from_supervisor,
    serialize_result,
    truncate_for_model,
)

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)


# State
class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str

    # Supervisor + guardrail
    guardrail_allowed:bool
    guardrail_reason:str
    selected_agents:list[str]
    trip_constraints:dict[str, Any]
    supervisor_reasoning:str

    # Original
    flight_results:str
    hotel_results:str
    weather_results:str
    itinerary:str

    # New budget + HITL state
    budget_results: str
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls:int

def supervisor_agent(state: TravelState):
    query = state["user_query"]
    llm_calls = state.get("llm_calls",0)

    guardrail_error = False
    try:
        guardrail_raw = llm_text(
            llm,
            render_prompt("guardrail_system"),
            render_prompt("guardrail_user", query=query),
        )

        guardrail_result = json_from_llm(guardrail_raw)
        allowed = guardrail_result.get("allowed") is True
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1

    except Exception as exc:
        print(f"Guardrail fallback used: {exc}")
        allowed = False
        guardrail_reason = ""
        guardrail_error = True

    if not allowed:
        reason = (
            "Sorry, I can help with travel planning and travel information, "
            "but I cannot book rides or arrange services."
            if guardrail_error
            else guardrail_reason
        ) or (
            "TripBuddy AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    try:
        supervisor_raw = llm_text(
            llm,
            render_prompt("supervisor_system"),
            render_prompt("supervisor_user", query=query),
        )
        parsed = json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        print(f"Supervisor fallback used: {exc}")
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = empty_constraints()
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }

def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


def flight_agent(state: TravelState):
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports", {}))
        airlines = asyncio.run(aviation_mcp_call("list_airlines", {}))

        prompt = render_prompt(
            "flight_user",
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(content=render_prompt("flight_system")),
                HumanMessage(content=prompt),
            ]
        )
        flight_data = response.content
    except Exception as exc:
        flight_data = f"Flight information unavailable: {exc}"

    return {
        "flight_results": flight_data,
        "messages": [AIMessage(content="Flight recommendations generated")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    # hotel_data = tavily_search(query)
    hotel_data = asyncio.run(tavily_mcp_search(query))

    return{
            "hotel_results": hotel_data,
            "messages":[AIMessage(content="Hotel information fetched.")],
            "llm_calls": state.get("llm_calls", 0) + 1
            }

# Weather Agent
def weather_agent(state: TravelState):
    city = extract_destination(
        state["user_query"],
        llm,
    )

    try:
        weather_data = asyncio.run(
            weather_mcp_call("get_current_weather", {"city": city})
        )

        forecast_data = asyncio.run(
            weather_mcp_call("get_forecast", {"city": city})
        )

        weather_results = f"""
                                Current Weather:
                                {weather_data}

                                Forecast:
                                {forecast_data}
                                """

    except Exception as exc:
        weather_results = (
            f"Live weather information for {city} "
            "is temporarily unavailable. Give general "
            "seasonal guidance and advise the traveler "
            "to verify the forecast before departure."
        )

    return {
        "weather_results": weather_results,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# Budget Agent
def budget_agent(state: TravelState):
    prompt = render_prompt(
        "budget_user",
        user_query=state["user_query"],
        trip_constraints=state.get("trip_constraints", {}),
        flight_results=state.get("flight_results", ""),
        hotel_results=state.get("hotel_results", ""),
        weather_results=state.get("weather_results", ""),
    )

    response = llm.invoke(
        [
            SystemMessage(content=render_prompt("budget_system")),
            HumanMessage(content=prompt),
        ]
    )

    return {
        "budget_results": response.content,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# Itenary Agent
def itinerary_agent(state: TravelState):
    flights_text = truncate_for_model(state.get("flight_results", ""), 1000)
    hotels_text = truncate_for_model(state.get("hotel_results", ""), 1000)
    weather_text = truncate_for_model(state.get("weather_results", ""), 900)
    budget_text = truncate_for_model(state.get("budget_results", ""), 900)

    prompt = render_prompt(
        "itinerary_user",
        user_query=state["user_query"],
        trip_constraints=state.get("trip_constraints", {}),
        flight_results=flights_text,
        hotel_results=hotels_text,
        weather_results=weather_text,
        budget_results=budget_text,
    )

    response = llm.invoke(
        [
            SystemMessage(content=render_prompt("itinerary_system")),
            HumanMessage(content=prompt),
        ]
    )

    approval_request = render_prompt("approval_request")

    return {
        "itinerary": response.content,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# HITL
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# Final Result Agent
def final_agent(state: TravelState):

    if state.get("approved", False):
        review_instruction = render_prompt(
            "final_review_approved"
        )
    else:
        review_instruction = render_prompt(
            "final_review_revision",
            feedback=state.get("human_feedback", "")
            or "Improve the draft before finalizing it.",
        )

    flights_text = truncate_for_model(state.get("flight_results", ""), 1000)
    hotels_text = truncate_for_model(state.get("hotel_results", ""), 1000)
    weather_text = truncate_for_model(state.get("weather_results", ""), 900)
    budget_text = truncate_for_model(state.get("budget_results", ""), 900)
    itinerary_text = truncate_for_model(
        state.get("itinerary") or state.get("itenary", ""),
        1800,
    )
    

    final_prompt = render_prompt(
        "final_user",
        review_instruction=review_instruction,
        user_query=state["user_query"],
        trip_constraints=state.get("trip_constraints", {}),
        flight_results=flights_text,
        hotel_results=hotels_text,
        weather_results=weather_text,
        budget_results=budget_text,
        itinerary=itinerary_text,
    )

    response = llm.invoke(
        [
            SystemMessage(content=render_prompt("final_system")),
            HumanMessage(content=final_prompt),
        ]
    )

    return {
        "final_response": response.content,
        "messages": [response],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }

# Dynamic Supervisor Routing
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


# Build graph
graph = StateGraph(TravelState)

graph.add_node("supervisor_agent", supervisor_agent)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("budget_agent", budget_agent)
graph.add_node("itinerary_agent", itinerary_agent)
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "supervisor_agent")
graph.add_conditional_edges("supervisor_agent",route_from_supervisor,ROUTE_MAP)
graph.add_conditional_edges("flight_agent", route_after_agent("flight_agent"), ROUTE_MAP)
graph.add_conditional_edges("hotel_agent", route_after_agent("hotel_agent"),ROUTE_MAP)
graph.add_conditional_edges("weather_agent", route_after_agent("weather_agent"),ROUTE_MAP)
graph.add_conditional_edges("budget_agent", route_after_agent("budget_agent"),ROUTE_MAP)
graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_edge("final_agent", END)
graph.add_edge("guardrail_blocked", END)


travel_graph = graph.compile(checkpointer=InMemorySaver())
def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "budget_results": "",
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")

    config = {"configurable": {"thread_id": thread_id}}
    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return serialize_result(result, thread_id)