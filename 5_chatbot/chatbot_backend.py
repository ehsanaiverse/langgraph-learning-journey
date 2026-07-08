from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from typing import TypedDict, Annotated
from dotenv import load_dotenv
import os
import sqlite3

load_dotenv()

GROQ_API_KEY=os.getenv("GROQ_API_KEY")
GROQ_MODEL=os.getenv("GROQ_MODEL")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)

# state define
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def chat_node(state: ChatState):
    # take user query from state
    messages = state["messages"]
    
    # send to llm
    response = llm.invoke(messages)
    
    # store in state
    return {"messages": [response]}


conn = sqlite3.connect(database="5_chatbot/chatbot.db", check_same_thread=False)


# checkpointer
checkpointer = SqliteSaver(conn=conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)

chatbot = graph.compile(checkpointer=checkpointer)


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

