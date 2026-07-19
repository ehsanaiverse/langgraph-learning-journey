from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_mcp_adapters.client import MultiServerMCPClient
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
import asyncio

# load the env file
load_dotenv()

# llm
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # moved out of source

if not GROQ_API_KEY or not GROQ_MODEL:
    raise RuntimeError("GROQ_API_KEY / GROQ_MODEL missing from env")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)


# define tools
search_tools = DuckDuckGoSearchResults()

# wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


# calculator + get_stock_price now live on MCP server (mcp_server.py), loaded at runtime below
 
# mcp client config: point at main.py over stdio
mcp_client = MultiServerMCPClient(
    {
        "calc_stock_server": {
            "command": "python3",
            "args": ["5_chatbot/mcp_server.py"],
            "transport": "stdio",
        }
    }
)


# state define
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    


async def build_graph():
    
    # fetch tools exposed by MCP server, merge with local tools
    mcp_tools = await mcp_client.get_tools()
    tools = [search_tools, wiki_tool] + mcp_tools

    # make the llm tool aware
    llm_with_tools = llm.bind_tools(tools)
    
    # define nodes
    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": [response]}


    tool_node = ToolNode(tools)

    # Graph Structure
    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)
    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()
    
    return chatbot


async def main():
    
    chatbot = await build_graph()
    
    response = await chatbot.ainvoke(
        {"messages": [HumanMessage(content="What is the stock price of Mango in Pakistan?")]}
    )
    
    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())