# backend.py

from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import interrupt, Command

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_groq.chat_models import ChatGroq
from langchain_core.tools import tool

from dotenv import load_dotenv
from typing import TypedDict, Annotated

import requests
import os

load_dotenv()

#configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY") 

# 1. LLM
if not GROQ_API_KEY or not GROQ_MODEL:
    raise RuntimeError("GROQ_API_KEY / GROQ_MODEL missing from env")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

# 2. Tools
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    url = (
        f"https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_INTRADAY&symbol={symbol}"
        f"&interval=5min&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": str(e)}


@tool
def purchase_stock(symbol: str, quantity: int) -> dict:
    """
    Simulate purchasing a given quantity of a stock symbol.

    HUMAN-IN-THE-LOOP:
    Before confirming the purchase, this tool will interrupt
    and wait for a human decision ("yes" / anything else).
    """
    # This pauses the graph and returns control to the caller
    decision = interrupt(f"Approve buying {quantity} shares of {symbol}? (yes/no)")
    
    try:
        if isinstance(decision, str) and decision.lower() == "yes":
            return {
                "status": "success",
                "message": f"Purchase order placed for {quantity} shares of {symbol}.",
                "symbol": symbol,
                "quantity": quantity,
            }
        
        else:
            return {
                "status": "cancelled",
                "message": f"Purchase of {quantity} shares of {symbol} was declined by human.",
                "symbol": symbol,
                "quantity": quantity,
            }
    except Exception as e:
        print(f"Error! {e}")


tools = [get_stock_price, purchase_stock]
llm_with_tools = llm.bind_tools(tools)


# 3. State
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# 4. Nodes
def chat_node(state: ChatState):
    """LLM node that may answer or request a tool call."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

tool_node = ToolNode(tools)


# 5. Checkpointer (in-memory)
memory = MemorySaver()


# 6. Graph
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=memory)


# 7. Simple usage example (CLI with HITL)
if __name__ == "__main__":
    
    # Use a fixed thread_id so the conversation is persisted in memory
    thread_id = "thread-1"

    while True:
        
        CONFIG = {"configurable": {"thread_id": thread_id}}
        
        user_input = input("You: ")
        if user_input.lower().strip() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Build initial state for this turn
        state = {"messages": [HumanMessage(content=user_input)]}

        # Run the graph (may hit an interrupt)
        result = chatbot.invoke(
            state,
            config=CONFIG,
        )

        # Check for HITL interrupt from purchase_stock
        interrupts = result.get("__interrupt__", [])

        if interrupts:
            # Our interrupt payload is the string we passed to interrupt(...)
            prompt_to_human = interrupts[0].value
            print(f"HITL: {prompt_to_human}")
            decision = input("Your decision: ").strip().lower()

            # Resume graph with the human decision ("yes" / "no" / whatever)
            result = chatbot.invoke(
                Command(resume=decision),
                config=CONFIG,
            )

        # Get the latest message from the assistant
        messages = result["messages"]
        last_msg = messages[-1]
        print(f"Bot: {last_msg.content}\n")