import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage


PROMPTS_PATH = Path(__file__).with_name("prompts.json")
with PROMPTS_PATH.open(encoding="utf-8") as prompts_file:
    PROMPTS: dict[str, str] = json.load(prompts_file)


KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def render_prompt(name: str, **values: object) -> str:
    """Load and interpolate named placeholders without interpreting JSON braces."""
    try:
        template = PROMPTS[name]
    except KeyError as exc:
        raise KeyError(f"Prompt '{name}' is not defined in {PROMPTS_PATH.name}.") from exc

    placeholders = set(re.findall(r"\{([a-zA-Z_]\w*)\}", template))
    missing = placeholders - values.keys()
    if missing:
        missing_names = ", ".join(sorted(missing))
        raise ValueError(f"Missing values for prompt '{name}': {missing_names}.")

    for key, value in values.items():
        template = template.replace(f"{{{key}}}", str(value))
    return template


def truncate_for_model(value: str | None, max_chars: int = 900) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n\n... [truncated for model token limit]"


def llm_text(llm: Any, system_prompt: str, user_prompt: str) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    return str(response.content)


def json_from_llm(text: str) -> dict[str, Any]:
    """Extract the first complete JSON object returned by the model."""
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise ValueError("The model did not return a JSON object.")

    parsed = json.loads(text[start : end + 1])
    if not isinstance(parsed, dict):
        raise ValueError("The model response JSON must be an object.")
    return parsed


def empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def extract_destination(query: str, llm: Any) -> str:
    prompt = render_prompt("destination_user", query=query)
    response = llm.invoke(
        [
            SystemMessage(content=render_prompt("destination_system")),
            HumanMessage(content=prompt),
        ]
    )
    destination = str(response.content).strip()
    if not destination:
        raise ValueError("The destination could not be extracted.")
    return destination


def selected_agents(state: dict[str, Any]) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: dict[str, Any]) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    if current_agent not in AGENT_ORDER:
        raise ValueError(f"Unknown travel agent '{current_agent}'.")

    def route(state: dict[str, Any]) -> str:
        agents = selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in agents:
                return next_agent

        return "itinerary_agent"

    return route


def interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def serialize_result(result: dict[str, Any], thread_id: str) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    answer = result.get("final_response") or last_message
    pending_interrupt = interrupt_payload(result)

    if pending_interrupt:
        answer = pending_interrupt.get("draft_itinerary") or result.get(
            "itinerary", ""
        )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": pending_interrupt is not None,
        "approval_request": (
            pending_interrupt.get("approval_request", "")
            if pending_interrupt
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": (
            pending_interrupt.get("draft_itinerary", "")
            if pending_interrupt
            else result.get("itinerary", "")
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }
