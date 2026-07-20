import streamlit as st
from chatbot_backend import chatbot, retrieve_all_threads, ingest_pdf, thread_has_document, thread_document_metadata
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
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
    st.session_state["chat_threads"] = retrieve_all_threads()

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

if st.session_state["thread_id"] not in st.session_state["chat_titles"]:
    st.session_state["chat_titles"][st.session_state["thread_id"]] = "New Chat"

add_thread_id(st.session_state["thread_id"])


# Sidebar UI
st.sidebar.title("LangGraph ChatBot")

if st.sidebar.button("New Chat"):
    reset_chat()

# --- PDF Upload ---
st.sidebar.header("Upload PDF")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    thread_id = str(st.session_state["thread_id"])

    with st.sidebar.status("Indexing PDF...", expanded=False) as status:
        try:
            result = ingest_pdf(file_bytes, thread_id, filename=uploaded_file.name)
            status.update(label=f"Indexed: {result['filename']}", state="complete")
            st.sidebar.success(
                f"Pages: {result['pages']} | Chunks: {result['chunks']}"
            )
        except Exception as e:
            status.update(label="Indexing failed", state="error")
            st.sidebar.error(str(e))

# Show current thread's document info
thread_id = str(st.session_state["thread_id"])
if thread_has_document(thread_id):
    meta = thread_document_metadata(thread_id)
    st.sidebar.info(
        f"Document loaded: **{meta.get('filename', 'unknown')}** "
        f"({meta.get('pages', 0)} pages, {meta.get('chunks', 0)} chunks)"
    )

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
        
    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_run"
        }
    
    with st.chat_message("assistant"):
        tool_info = st.empty()
        response_placeholder = st.empty()
        full_response = ""
        tools_used = []

        try:
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages"
            ):
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "unknown_tool")
                    tool_content = message_chunk.content
                    tools_used.append(tool_name)

                    tool_info.markdown(
                        f"🔧 **`{tool_name}`** → `{tool_content[:80]}{'...' if len(tool_content) > 80 else ''}`"
                    )

                elif message_chunk.content:
                    full_response += message_chunk.content
                    response_placeholder.markdown(full_response + "▌")
        except Exception as e:
            full_response = f"Error: {e}"
            response_placeholder.error(full_response)

        if tools_used:
            tool_names = ", ".join(f"`{t}`" for t in tools_used)
            tool_info.markdown(f"✅ Tools used: {tool_names}")

        response_placeholder.markdown(full_response)

    st.session_state["message_history"].append({"role": "assistant", "content": full_response})

    if st.session_state["chat_titles"][st.session_state["thread_id"]] == "New Chat":
        title = user_input[:40] + ("..." if len(user_input) > 40 else "")
        st.session_state["chat_titles"][st.session_state["thread_id"]] = title