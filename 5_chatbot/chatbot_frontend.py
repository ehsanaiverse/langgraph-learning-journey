import streamlit as st
from chatbot_backend import chatbot
from langchain_core.messages import HumanMessage
import uuid

# utility functions
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id
    add_thread_id(st.session_state["thread_id"])
    st.session_state["message_history"] = []
    st.session_state["chat_titles"][thread_id] = "New Chat"

def add_thread_id(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)
        
def load_coversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])       


# session setup
# st.session_state -> dict
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = []

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

if st.session_state["thread_id"] not in st.session_state["chat_titles"]:
    st.session_state["chat_titles"][st.session_state["thread_id"]] = "New Chat"

add_thread_id(st.session_state["thread_id"])


# Sidebar UI
st.sidebar.title("LangGraph ChatBot")

if st.sidebar.button("New Chat"):
    reset_chat()
    
st.sidebar.header("Convesation History")

for thread_id in st.session_state["chat_threads"][::-1]:
    title = st.session_state["chat_titles"].get(thread_id, str(thread_id))
    if st.sidebar.button(title, key=str(thread_id)):
        st.session_state["thread_id"] = thread_id
        messages = load_coversation(thread_id)
        
        temp_messages = []
        
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"
            
            temp_messages.append({"role": role, "content": msg.content})
        st.session_state["message_history"] = temp_messages

# main UI

# loading the conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# [{"role": "user", "content": 'Hi'}
# {"role": "assistant", "content": 'Hello'}]

# user input
user_input = st.chat_input("Type here...")

if user_input:
    
    # first add the message to message history
    st.session_state["message_history"].append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.text(user_input)
        
    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}
    
    # first add the message to message history
    with st.chat_message("assistant"):
        
        ai_message = st.write_stream(
            message_chunk.content for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
                )
            )

    st.session_state["message_history"].append({"role": "assistant", "content": ai_message})

    if st.session_state["chat_titles"][st.session_state["thread_id"]] == "New Chat":
        title = user_input[:40] + ("..." if len(user_input) > 40 else "")
        st.session_state["chat_titles"][st.session_state["thread_id"]] = title