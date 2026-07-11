from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.tools import tool
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
import sqlite3
import requests

# load the env file
load_dotenv()

# llm
GROQ_API_KEY=os.getenv("GROQ_API_KEY")
GROQ_MODEL=os.getenv("GROQ_MODEL")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)


# define tools
search_tools = DuckDuckGoSearchResults()

# calculator tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic aurtimatic operation on two number.
    Supposrted operations are: add, sub, mul, div
    """    
    try:
        if(operation=="add"):
            result = first_num + second_num
        if(operation=="sub"):
            result = first_num - second_num
        if(operation=="mul"):
            result = first_num * second_num
        if(operation=="div"):
            if(second_num == 0):
                return {"error": "zero division error"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation {operation}"}
        
        return {"first_num": first_num, "second_num": second_num,"result": result}
    except Exception as e:
        return {"error": str(e)}

# stuck price tool
@tool
def get_stuck_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA') 
    using Alpha Vantage with API key in the URL.
    """
    
    url: f"https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey=XQ6EDBDL3WPWN7UX"
    response = requests.get(url)
    
    return response.json()


# make tool list
tools = [calculator, search_tools, get_stuck_price]

# make the llm tools awar
llm_with_tools = llm.bind_tools(tools)


# state define
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# define nodes
def chat_node(state: ChatState):
    # take user query from state
    messages = state["messages"]
    # send to llm
    response = llm_with_tools.invoke(messages)
    # store in state
    return {"messages": [response]}


# excutes tool calls
tool_node = ToolNode(tools)


# connect to database
conn = sqlite3.connect(database="5_chatbot/chatbot.db", check_same_thread=False)


# checkpointer
checkpointer = SqliteSaver(conn=conn)


# Graph Structure
graph = StateGraph(ChatState)

# graph nodes
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# graph edges
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

# compile the workflow
chatbot = graph.compile(checkpointer=checkpointer)


# helper fuction
def retrieve_all_threads():
    # extract unique thread
    all_threads = set()
    # list of all threads existing in db
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_threads)


# Streaming feature

# for message_chunk, metadata in chatbot.stream(
#     {"messages": [HumanMessage(content="user_input")]},
#     config={"configurable": {"thread_id": "thread_1"}},
#     stream_mode="messages"
# ):
#     if message_chunk.content:
#         print(message_chunk.content, end=" ", flush=True)



# CONFIG={"configurable": {"thread_id": "thread-1"}}
# response = chatbot.invoke(
#     {"messages": [HumanMessage(content="What is my name?")]},
#     config=CONFIG
# )

# print(chatbot.get_state(config=CONFIG).values["messages"])

# print(response)

