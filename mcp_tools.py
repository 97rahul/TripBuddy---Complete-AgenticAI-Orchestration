import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

SERVERS = {
    "tavily": {
        "transport": "streamable_http",
        "url": (
            "https://mcp.tavily.com/mcp/"
            f"?tavilyApiKey={TAVILY_API_KEY or ''}"
        ),
    },
    "aviationstack": {
        "transport": "stdio",
        "command": "uvx",
        "args": ["aviationstack-mcp"],
        "env": (
            {"AVIATIONSTACK_API_KEY": AVIATIONSTACK_API_KEY}
            if AVIATIONSTACK_API_KEY
            else {}
        ),
    },
    "weather": {
        "transport": "stdio",
        "command": sys.executable,
        "args": [str(BASE_DIR / "weather_mcp_tool.py")],
        "env": (
            {"OPENWEATHER_API_KEY": OPENWEATHER_API_KEY}
            if OPENWEATHER_API_KEY
            else {}
        ),
    },
}

CLIENTS = {
    server_name: MultiServerMCPClient({server_name: server_config})
    for server_name, server_config in SERVERS.items()
}
_tool_cache = {}


async def _get_server_tools(server_name: str):
    if server_name not in CLIENTS:
        raise ValueError(f"Unknown MCP server '{server_name}'.")

    if server_name not in _tool_cache:
        tools = await CLIENTS[server_name].get_tools()
        _tool_cache[server_name] = {tool.name: tool for tool in tools}

    return _tool_cache[server_name]


async def tavily_mcp_search(query: str):
    tools = await _get_server_tools("tavily")
    tool = tools.get("tavily_search")
    if tool is None:
        raise ValueError("Tavily MCP server did not provide 'tavily_search'.")

    return await tool.ainvoke({"query": query})


async def aviation_mcp_call(tool_name: str, tool_args: dict):
    tools = await _get_server_tools("aviationstack")
    tool = tools.get(tool_name)
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found.")

    return await tool.ainvoke(tool_args or {})


async def weather_mcp_call(tool_name: str, tool_args: dict):
    tools = await _get_server_tools("weather")
    tool = tools.get(tool_name)
    if tool is None:
        raise ValueError(f"Tool '{tool_name}' not found.")

    return await tool.ainvoke(tool_args or {})
