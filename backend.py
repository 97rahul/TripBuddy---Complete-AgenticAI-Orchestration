from typing import TypedDict, Annotated
import operator
import uuid
import os
from tools.flight_tool import search_flights
from tools.tavily_tool import tavily_search
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage,AnyMessage
from langgraph.graph import StateGraph,START,END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()


def truncate_for_model(value: str, max_chars: int = 900) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n... [truncated for model token limit]"


llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=os.getenv("GROQ_API_KEY")
)

# State
class TravelState(TypedDict):
    messages:Annotated[list[AnyMessage],operator.add]
    user_query:str
    flights_results:str
    hotel_results:str
    itenary:str
    llm_calls:int


# Flight Agent
def flight_agent(state: TravelState):
    query = state["user_query"]
    flight_data = search_flights(query)

    return{
            "flights_results": flight_data,
            "messages":[AIMessage(content="Flight results fetched.")],
            "llm_calls": state.get("llm_calls", 0) + 1
            }


# Hotel Agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_data = tavily_search(query)

    return{
            "hotel_results": hotel_data,
            "messages":[AIMessage(content="Hotel information fetched.")],
            "llm_calls": state.get("llm_calls", 0) + 1
            }

# Itenary Agent
def itenary_agent(state: TravelState):
    flights_text = truncate_for_model(state.get("flights_results", ""), 700)
    hotels_text = truncate_for_model(state.get("hotel_results", ""), 700)

    prompt = f"""
            Create a concise but useful travel itinerary.

            User Query:
            {state['user_query']}

            Flight Results:
            {flights_text}

            Hotel Results:
            {hotels_text}

            Keep it practical, budget-aware, and easy to follow.
            """

    response = llm.invoke([
        SystemMessage(content="You are an expert travel planner."),
        HumanMessage(content=prompt)
        ])

    return{
            "itenary": response.content,
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
            }


# Final Result Agent
def final_agent(state: TravelState):
    flights_text = truncate_for_model(state.get("flights_results", ""), 600)
    hotels_text = truncate_for_model(state.get("hotel_results", ""), 600)
    itinerary_text = truncate_for_model(state.get("itenary", ""), 900)

    final_prompt = f"""
                    Produce a concise final travel plan.

                    User Query:
                    {state['user_query']}

                    Flight Results:
                    {flights_text}

                    Hotel Results:
                    {hotels_text}

                    Itinerary:
                    {itinerary_text}

                    Use these sections:
                    1. Trip Summary
                    2. Flight Information
                    3. Hotel Suggestions
                    4. Day-by-Day Itinerary
                    5. Estimated Budget
                    6. Final Recommendations

                    Keep it practical and concise. Mention that live flight API may not show actual ticket prices if unavailable.
                    """

    response = llm.invoke([
        SystemMessage(content="You are a professional AI travel booking assistant."),
        HumanMessage(content=final_prompt)
        ])

    return{
            "messages": [response],
            "llm_calls": state.get("llm_calls", 0) + 1
            }


# Build graph

graph = StateGraph(TravelState)
graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("itenary_agent", itenary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "itenary_agent")
graph.add_edge("itenary_agent", "final_agent") 
graph.add_edge("final_agent", END)


travel_graph = graph.compile(checkpointer=InMemorySaver())


def run_travel_agent(user_input:str,thread_id:str | None = None):

    if not thread_id:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "flights_results": "",
            "hotel_results": "",
            "itenary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    itinerary_value = result.get("itenary") or result.get("itinerary") or ""
    flight_value = result.get("flights_results") or result.get("flight_results") or ""
    hotel_value = result.get("hotel_results") or result.get("hotels_results") or ""

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": flight_value,
        "flights_results": flight_value,
        "hotel_results": hotel_value,
        "hotels_results": hotel_value,
        "itinerary": itinerary_value,
        "itenary": itinerary_value,
        "llm_calls": result.get("llm_calls", 0),
        "success": True,
    }
