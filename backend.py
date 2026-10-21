import asyncio
from typing import TypedDict, Annotated
import operator
import uuid
import os
# from tools.flight_tool import search_flights
# from tools.tavily_tool import tavily_search
from mcp_tools import tavily_mcp_search,aviation_mcp_call, weather_mcp_call
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
    flight_results:str
    hotel_results:str
    weather_results:str
    itenary:str
    llm_calls:int


# # Flight Agent
# def flight_agent(state: TravelState):
#     query = state["user_query"]
#     flight_data = search_flights(query)

#     return{
#             "flight_results": flight_data,
#             "messages":[AIMessage(content="Flight results fetched.")],
#             "llm_calls": state.get("llm_calls", 0) + 1
#             }

FLIGHT_AGENT_PROMPT = """
You are a travel flight expert.

User Query:
{query}

Airport Information:
{airport_data}

Airline Information:
{airline_data}

Generate:
1. Likely departure airport
2. Likely arrival airport
3. Airlines serving this route
4. Typical flight duration
5. Estimated airfare range
6. Peak season pricing warning
7. Booking advice

Return concise travel guidance.
"""


def flight_agent(state: TravelState):
    print("\nINSIDE FLIGHT AGENT\n")
    query = state["user_query"]

    try:
        airports = asyncio.run(aviation_mcp_call("list_airports"))
        airlines = asyncio.run(aviation_mcp_call("list_airlines"))

        print("\nAIRPORTS:", airports)
        print("\nAIRLINES:", airlines)

        prompt = FLIGHT_AGENT_PROMPT.format(
            query=query,
            airport_data=str(airports)[:3000],
            airline_data=str(airlines)[:3000],
        )

        response = llm.invoke(
            [
                SystemMessage(content="You are an expert travel flight planner."),
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

def extract_destination(query: str) -> str:
    prompt = f"""
            Extract only the destination city or country from the travel request.

            Travel request:
            {query}

            Return only the destination name.
            Do not add any explanation.
            """

    response = llm.invoke(prompt)

    destination = str(
        response.content
    ).strip()

    if not destination:
        raise ValueError(
            "The destination could not be extracted."
        )

    return destination

# Weather Agent
def weather_agent(state: TravelState):
    city = extract_destination(
        state["user_query"]
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
        print(
            f"WEATHER AGENT MCP ERROR: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )

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

# Itenary Agent
def itenary_agent(state: TravelState):
    flights_text = truncate_for_model(state.get("flight_results", ""), 700)
    hotels_text = truncate_for_model(state.get("hotel_results", ""), 700)
    weather_text = truncate_for_model(state.get("weather_results", ""), 700)

    prompt = f"""
            Create a concise but useful travel itinerary.

            User Query:
            {state['user_query']}

            Flight Results:
            {flights_text}

            Hotel Results:
            {hotels_text}

            Weather Results:
            {weather_text}

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
    flights_text = truncate_for_model(state.get("flight_results", ""), 600)
    hotels_text = truncate_for_model(state.get("hotel_results", ""), 600)
    itinerary_text = truncate_for_model(state.get("itenary", ""), 900)
    weather_text = truncate_for_model(state.get("weather_results", ""), 700)

    final_prompt = f"""
                    Produce a concise final travel plan.

                    User Query:
                    {state['user_query']}

                    Flight Results:
                    {flights_text}

                    Hotel Results:
                    {hotels_text}

                    Weather Results:
                    {weather_text}

                    Itinerary:
                    {itinerary_text}

                    Use these sections:
                    1. Trip Summary
                    2. Flight Information
                    3. Hotel Suggestions
                    4. Weather Information
                    5. Day-by-Day Itinerary
                    6. Estimated Budget
                    7. Final Recommendations

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
graph.add_node("weather_agent", weather_agent)
graph.add_node("itenary_agent", itenary_agent)
graph.add_node("final_agent", final_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent", "itenary_agent")
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
            "flight_results": "",
            "hotel_results": "",
            "weather_results": "",
            "itenary": "",
            "llm_calls": 0
        },
        config=config
    )

    final_answer = result["messages"][-1].content

    itinerary_value = result.get("itenary") or result.get("itinerary") or ""
    flight_value = result.get("flight_results") or result.get("flights_results") or ""
    hotel_value = result.get("hotel_results") or result.get("hotels_results") or ""
    weather_value = result.get("weather_results") or result.get("weathers_results") or ""

    return {
        "thread_id": thread_id,
        "answer": final_answer,
        "flight_results": flight_value,
        "hotel_results": hotel_value,
        "weather_results": weather_value,
        "itenary": itinerary_value,
        "llm_calls": result.get("llm_calls", 0),
        "success": True,
    }
