# TripBuddy

TripBuddy is an AI-assisted travel-planning application. It can research travel
information and prepare a budget-aware trip plan with flights, accommodation,
weather, and an itinerary. Review the draft and either approve it or submit
feedback before receiving the final response.

TripBuddy provides information only. It does not book flights, hotels, rides, or
other travel services. Prices and budget estimates may be approximate and
should be independently verified before booking.

## Features

- Natural-language trip planning, including destinations, duration, origin,
  budget, and preferences.
- Specialist planning for flights, hotels, weather, budget, and itinerary.
- A request guardrail for unrelated or unsafe requests and unsupported service
  bookings.
- Human review of the draft itinerary, with optional feedback to guide the final
  response.
- A FastAPI web application with a downloadable itinerary PDF.
- An optional Streamlit interface for the same planning workflow.

## Requirements

- Python 3.10 or later.
- API credentials for the services you want to use (see
  [Configuration](#configuration)).
- `uvx` available on `PATH` for the AviationStack MCP server used by flight
  research.
- An internet connection for the model and external travel-data providers.

Python packages are listed in [`requirements.txt`](./requirements.txt).

## Setup

Run these commands from the project directory in PowerShell:

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell prevents environment activation, either enable script activation
for your user or invoke the environment's Python directly:

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the project root. Add the keys for the providers you
intend to use:

```dotenv
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
OPENWEATHER_API_KEY=your_openweather_api_key

# Optional; defaults to 8001 for the FastAPI application.
TRIPBUDDY_PORT=8001
```

`GROQ_API_KEY` is used by the language model. The other keys power hotel/web
search, aviation data, and weather respectively. Keep `.env` private; do not
commit API keys or share them in logs or screenshots. Missing credentials or
provider outages can make the corresponding live information unavailable.

## Run the application

### FastAPI web app

From the project directory, with the virtual environment active:

```powershell
python app.py
```

Open [http://127.0.0.1:8001](http://127.0.0.1:8001). To use another port, set
`TRIPBUDDY_PORT` before starting the server:

```powershell
$env:TRIPBUDDY_PORT = "8002"
python app.py
```

The server binds to `127.0.0.1` by default. If the chosen port is already in
use, stop the existing TripBuddy process or choose another port.

### Streamlit app

In a separate terminal:

```powershell
streamlit run streamlit_app.py
```

Streamlit prints the local URL to open in your browser. The FastAPI server is
not required for this interface; both interfaces call the backend workflow
directly.

## Using the planner

1. Describe your trip, including details such as origin, destination, dates or
   duration, number of travelers, and total budget.
2. Generate the plan and review the draft itinerary.
3. Approve the draft, or provide feedback and choose **Request changes**.
4. Read the final response and verify important details, availability, and
   prices with providers before making bookings.

For example:

> Plan a 7-day trip to Japan from Delhi for two people, under ₹1,50,000 total.
> Include budget stays, local transport suggestions, weather, and a day-by-day
> itinerary.

TripBuddy may provide approximate cost estimates rather than live fares or
rates. A requested revision is applied when preparing the final response; it
does not guarantee that the revised budget is achievable or rebook any service.

## HTTP API

The FastAPI application exposes:

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Serve the web interface |
| `GET` | `/health` | Basic server health check |
| `POST` | `/api/travel` | Start a trip-planning run |
| `POST` | `/api/travel/approval` | Submit approval or feedback for a draft |

Start a plan:

```json
{
  "message": "Plan a 5-day trip to Kyoto from Mumbai under ₹90,000."
}
```

The response includes an `answer`, `thread_id`, specialist results such as
`flight_results`, `hotel_results`, `weather_results`, and `budget_results`, plus
`requires_approval`. When `requires_approval` is `true`, use the returned
`thread_id` to submit the review:

```json
{
  "thread_id": "thread-id-from-the-start-response",
  "approved": false,
  "feedback": "Keep the total for two travelers under ₹90,000 and prioritize lower-cost stays."
}
```

Set `approved` to `true` to approve the draft. For revisions, set it to `false`
and describe the desired changes in `feedback`. The approval endpoint resumes
the existing workflow; submit the `thread_id` returned by the planning
endpoint.

## Project layout

| File or directory | Responsibility |
| --- | --- |
| `app.py` | FastAPI application, HTML serving, health check, and planning/review API |
| `streamlit_app.py` | Streamlit user interface |
| `backend.py` | LangGraph state, travel agents, routing, draft approval, and public workflow functions |
| `utils.py` | Shared prompt, parsing, routing, and result-serialization helpers |
| `prompts.json` | Guardrail, supervisor, specialist, approval, and final-response prompts |
| `mcp_tools.py` | MCP clients for Tavily, AviationStack, and the local weather server |
| `weather_mcp_tool.py` | OpenWeather MCP tools for current conditions and forecast |
| `templates/index.html` | FastAPI web interface and PDF generation |
| `tests/` | Backend, API, MCP, and utility tests |

When modifying the workflow, keep agent orchestration in `backend.py`, shared
non-agent helpers in `utils.py`, and prompt wording in `prompts.json`.

## Tests

Install the project dependencies, then run:

```powershell
python -m pytest tests -q
```

Or use the virtual environment directly:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```

## Troubleshooting

- **Port already in use:** another server is listening on the configured port.
  Stop that process or choose another `TRIPBUDDY_PORT`.
- **Model/API errors:** check that `GROQ_API_KEY` is set in the project `.env`
  and that the provider is reachable.
- **Flight data unavailable:** verify `AVIATIONSTACK_API_KEY` and that `uvx` is
  installed and available on `PATH`.
- **Hotel search unavailable:** verify `TAVILY_API_KEY`.
- **Weather unavailable:** verify `OPENWEATHER_API_KEY`. Weather results are
  retrieved by the local MCP server in `weather_mcp_tool.py`.
- **Review could not resume:** use the exact `thread_id` from the draft
  response and keep the server process running between planning and review.
