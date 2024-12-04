from utils.tools import get_web_docs, get_retriever_tool, get_prebuilt_retriever_tool, web_search_tool
import os
from dotenv import load_dotenv
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.vectorstores import FAISS
from utils.states import AgentState, print_messages_state
from langgraph.graph import StateGraph
from utils.llms import get_ollama, get_ollama_embedding
from langgraph.checkpoint.memory import MemorySaver
from utils.graphs import get_react_agent
import faiss

load_dotenv()

embeddings = get_ollama_embedding("all-minilm:l6-v2", os.getenv("BASE_URL"))
llm = get_ollama("llama3.1", os.getenv("BASE_URL"))

# # Create embeddings for the documents
doc_query = "medicines"
# docs = get_web_docs(doc_query)
# tools = [get_retriever_tool(doc_query, docs, embeddings, InMemoryVectorStore)]

# using prebuilt vector store
vectorstore = FAISS.load_local(
    "products", embeddings, allow_dangerous_deserialization=True
)

tools = [get_prebuilt_retriever_tool(doc_query, vectorstore)]

workflow = StateGraph(AgentState)

app = get_react_agent(workflow, "You are a helpful AI assistant, please respond to the users query to the best of your ability!",
                       llm, tools).compile(
    checkpointer=MemorySaver(),
)

config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# print(app.invoke({"messages": [("human", "Can you please give me the name of the capsules that are an inhibitor of the enzyme carbonic anhydrase, give me only the name")]}, config=config))
# print_messages_state(app.get_state(config=config))


# orally once daily with or without food 
print(app.invoke({"messages": [("human", "Can you provide dosage details of atorvastatin calcium tablets")]}, config=config))

