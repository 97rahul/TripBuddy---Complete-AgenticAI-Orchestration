from langchain_mcp_adapters.client import MultiServerMCPClient
from dotenv import load_dotenv
import os
import asyncio
import subprocess

load_dotenv()

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
            "args": [
                "aviationstack-mcp",
            ],
            "env": {"AVIATIONSTACK_API_KEY": AVIATIONSTACK_API_KEY}
        },
    "weather": {
            "transport": "stdio",
            "command": "python",
            # Uses the weather server inside the current project folder.
            "args": [
                r"C:\Users\agarw\OneDrive\Desktop\TripBuddy\TripBuddy---Complete-AgenticAI-Orchestration\weather_mcp_tool.py"
            ],

            "env":
                {OPENWEATHER_API_KEY: OPENWEATHER_API_KEY},
        },
}


client = MultiServerMCPClient(SERVERS)


search_tool = None
aviation_tools = {}
weather_tools = {}
async def initialize_mcp():

    global search_tool
    global aviation_tools
    global weather_tools

    if search_tool is not None and aviation_tools and weather_tools:
        return

    tools = await client.get_tools()

    for tool in tools:
        print(tool.name)

    search_tool = next(
        (tool for tool in tools if tool.name == "tavily_search")
    )

    aviation_tools = {
        tool.name: tool for tool in tools if tool.name!="tavily_search"
    }

    weather_tools = {
        tool.name: tool for tool in tools if tool.name!="tavily_search" and tool.name!="aviationstack"
    }

async def tavily_mcp_search(query: str):
    await initialize_mcp()
    
    result = await search_tool.ainvoke(
        {
            "query": query,
        }
    )
    print(result)
    return result



async def aviation_mcp_call(tool_name:str,tool_args:dict):

    await initialize_mcp()

    tool = aviation_tools.get(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found.")

    result = await tool.ainvoke(tool_args or {})
    print(result)
    return result


async def weather_mcp_call(tool_name:str,tool_args:dict):

    await initialize_mcp()

    tool = weather_tools.get(tool_name)
    if not tool:
        raise ValueError(f"Tool '{tool_name}' not found.")

    result = await tool.ainvoke(tool_args or {})
    print(result)
    return result