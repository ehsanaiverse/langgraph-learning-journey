from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_community.tools import DuckDuckGoSearchResults, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import tool
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

from typing import TypedDict, Annotated, Dict, Any, Optional
from dotenv import load_dotenv

import os
import sqlite3
import requests
import tempfile

# load the env file
load_dotenv()

# configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")  # moved out of source
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
HUGGINGFACE_AMBEDDING_MODEL=os.getenv("HUGGINGFACE_AMBEDDING_MODEL")

if not HUGGINGFACE_AMBEDDING_MODEL or not HUGGINGFACE_API_KEY:
    raise RuntimeError("HUGGINGFACE_API_KEY / HUGGINGFACE_AMBEDDING_MODEL missing from env — needed for PDF embeddings")

os.environ["HUGGING_FACE_HUB_TOKEN"] = HUGGINGFACE_API_KEY
embeddings = HuggingFaceEmbeddings(model=HUGGINGFACE_AMBEDDING_MODEL)


if not GROQ_API_KEY or not GROQ_MODEL:
    raise RuntimeError("GROQ_API_KEY / GROQ_MODEL missing from env")

llm = ChatGroq(model=GROQ_MODEL, api_key=GROQ_API_KEY)


# --- NEW: Per-thread storage (in-memory for now) ---
_THREAD_RETRIEVERS: Dict[str, Any] = {}      # thread_id → retriever
_THREAD_METADATA: Dict[str, dict] = {}       # thread_id → doc info



def _get_retriever(thread_id: Optional[str]):
    """Fetch the retriever for a thread if available."""
    if thread_id and thread_id in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[thread_id]
    return None


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: Optional[str] = None) -> dict:
    """
    Turn a PDF into a searchable knowledge base for one chat thread.
    
    Steps:
    1. Save bytes to temp file
    2. Load & split into chunks
    3. Convert chunks to vectors (embeddings)
    4. Store in FAISS index
    5. Return summary stats
    """
    if not file_bytes:
        raise ValueError("No PDF bytes received")

    # Step 1: Write to temp file (PyPDFLoader needs a file path)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        # Step 2a: Load PDF pages
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()  # List of Document objects (one per page)
        
        # Step 2b: Split pages into smaller chunks
        # Why? LLMs have token limits; we want precise retrieval
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,        # Each chunk ~1000 characters
            chunk_overlap=200,      # Overlap prevents losing context at boundaries
            separators=["\n\n", "\n", " ", ""]  # Prefer splitting at paragraphs
        )
        chunks = splitter.split_documents(documents)
        
        # Step 3 & 4: Create vector store from chunks
        # FAISS = Facebook AI Similarity Search (fast nearest-neighbor search)
        vector_store = FAISS.from_documents(chunks, embeddings)
        
        # Create retriever: given a question, returns top 4 similar chunks
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 4}
        )
        
        # Store for this thread
        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename or os.path.basename(tmp_path),
            "pages": len(documents),
            "chunks": len(chunks),
        }
        
        return {
            "status": "success",
            "filename": filename or os.path.basename(tmp_path),
            "pages": len(documents),
            "chunks": len(chunks),
        }
        
    finally:
        # Clean up temp file (FAISS keeps the text in memory)
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# define tools
search_tools = DuckDuckGoSearchResults()

# wikipedia tool
api_wrapper = WikipediaAPIWrapper(top_k_results=3, doc_content_chars_max=1000)
wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)


# calculator tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform basic arithmetic operation on two numbers.
    Supported operations: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "zero division error"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation {operation}"}

        return {"first_num": first_num, "second_num": second_num, "result": result}
    except Exception as e:
        return {"error": str(e)}


# stock price tool
@tool
def get_stuck_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a given symbol (e.g. 'AAPL', 'TSLA')
    using Alpha Vantage.
    """
    if not ALPHA_VANTAGE_API_KEY:
        return {"error": "ALPHA_VANTAGE_API_KEY not set"}

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
def rag_tool(query: str, thread_id: Optional[str] = None) -> dict:
    """
    Retrieve relevant information from the pdf document.
    Use this tool when the user asks factual / conceptual questions
    that might be answered from the stored documents.
    """
    
    retriever = _get_retriever(thread_id)
    if retriever is None:
        return {
            "error": "No document indexed for this chat. Upload a PDF first.",
            "query": query,
        }    
    
    result = retriever.invoke(query)
    
    content = [doc.page_content for doc in result]
    metadata = [doc.metadata for doc in result]
    
    return {
        "query": query,
        "content": content,
        "metadata": metadata
    }


# make tool list
tools = [search_tools, wiki_tool, calculator, get_stuck_price, rag_tool]

# make the llm tool aware
llm_with_tools = llm.bind_tools(tools)


# state define
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# define nodes
def chat_node(state: ChatState, config=None):
    """LLM node that may answer or request a tool call."""
    thread_id = None
    if config and isinstance(config, dict):
        thread_id = config.get("configurable", {}).get("thread_id")

    system_message = SystemMessage(
        content=(
            "You are a helpful assistant. For questions about the uploaded PDF, call "
            "the `rag_tool` and include the thread_id "
            f"`{thread_id}`. You can also use the web search, stock price, and "
            "calculator tools when helpful. If no document is available, ask the user "
            "to upload a PDF."
        )
    )

    messages = [system_message, *state["messages"]]
    response = llm_with_tools.invoke(messages, config=config)
    return {"messages": [response]}


tool_node = ToolNode(tools, handle_tool_errors=True)

# connect to database (ensure dir exists)
os.makedirs("5_chatbot", exist_ok=True)
conn = sqlite3.connect(database="5_chatbot/chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=conn)

# Graph Structure
graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)
graph.add_edge(START, "chat_node")
graph.add_conditional_edges("chat_node", tools_condition)
graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)


def retrieve_all_threads():
    all_threads = set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config["configurable"]["thread_id"])
    return list(all_threads)


def thread_has_document(thread_id: str) -> bool:
    return str(thread_id) in _THREAD_RETRIEVERS


def thread_document_metadata(thread_id: str) -> dict:
    return _THREAD_METADATA.get(str(thread_id), {})


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

