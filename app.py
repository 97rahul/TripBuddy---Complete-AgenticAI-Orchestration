import traceback
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import resume_travel_agent, run_travel_agent

BASE_DIR = Path(__file__).resolve().parent

app = FastAPI(
    title="TripBuddy - Your AI Travel Companion",
    description="TripBuddy is an AI-powered travel assistant that helps users plan trips with flight, hotel, and weather information.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


class TravelRequest(BaseModel):
    message: str
    thread_id: str | None = None


class TravelApprovalRequest(BaseModel):
    thread_id: str
    approved: bool
    feedback: str = ""


def travel_response(result: dict[str, Any]) -> dict[str, Any]:
    itinerary = result.get("itinerary") or result.get("itenary", "")
    return {
        "success": True,
        "thread_id": result["thread_id"],
        "answer": result["answer"],
        "flight_results": result.get("flight_results", ""),
        "hotel_results": result.get("hotel_results", ""),
        "weather_results": result.get("weather_results", ""),
        "budget_results": result.get("budget_results", ""),
        "itinerary": itinerary,
        "itenary": itinerary,
        "requires_approval": result.get("requires_approval", False),
        "approval_request": result.get("approval_request", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))


@app.post("/api/travel")
def travel_planner(request: TravelRequest):
    try:
        user_message = request.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={"success": False, "error": "Message cannot be empty."},
            )

        result = run_travel_agent(user_message, request.thread_id)

        return JSONResponse(content=travel_response(result))

    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()

        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": (
                    "The trip planner could not complete this request. "
                    "Please try again shortly."
                ),
            },
        )


@app.post("/api/travel/approval")
def travel_approval(request: TravelApprovalRequest):
    thread_id = request.thread_id.strip()
    if not thread_id:
        return JSONResponse(
            status_code=400,
            content={"success": False, "error": "A thread_id is required."},
        )

    try:
        result = resume_travel_agent(
            thread_id,
            request.approved,
            request.feedback,
        )
        return JSONResponse(content=travel_response(result))
    except Exception as e:
        print("ERROR:", e)
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": (
                    "The trip planner could not resume this request. "
                    "Please try again shortly."
                ),
            },
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("TRIPBUDDY_PORT", "8001"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
