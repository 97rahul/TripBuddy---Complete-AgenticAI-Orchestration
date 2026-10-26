# TripBuddy

TripBuddy is an AI travel-planning application with FastAPI and Streamlit interfaces.

## Backend layout

- `backend.py` defines the LangGraph state, agent nodes, graph wiring, and public
  `run_travel_agent` / `resume_travel_agent` entry points.
- `utils.py` contains shared prompt loading/rendering, model-response parsing,
  state helpers, agent routing, and result serialization.
- `prompts.json` stores system and user prompt templates. Use `{placeholder}`
  names for values supplied by `render_prompt`.
- `mcp_tools.py` provides the MCP service clients used by the travel agents.

Keep reusable, non-agent logic in `utils.py`; keep agent orchestration in
`backend.py`; and keep prompt wording in `prompts.json`.

## Tests

Run the test suite with:

```powershell
.\venv\Scripts\python.exe -m pytest tests -q
```