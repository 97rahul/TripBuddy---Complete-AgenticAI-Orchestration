import traceback
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend import run_travel_agent

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
        itinerary = result.get("itinerary") or result.get("itenary", "")

        return JSONResponse(
            content={
                "success": True,
                "thread_id": result["thread_id"],
                "answer": result["answer"],
                "flight_results": result["flight_results"],
                "hotel_results": result["hotel_results"],
                "weather_results": result["weather_results"],
                "itinerary": itinerary,
                "itenary": itinerary,
                "llm_calls": result["llm_calls"],
            }
        )

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


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("TRIPBUDDY_PORT", "8001"))
    uvicorn.run("app:app", host="127.0.0.1", port=port, reload=False)
