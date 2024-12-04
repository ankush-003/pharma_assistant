from utils.llms import get_ollama_embedding, get_ollama
from utils.states import AgentState, print_messages_state
from utils.tools import web_search_tool, get_prebuilt_retriever_tool
from utils.graphs import stream_graph_updates
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from utils.prompts import SYSTEM_PROMPTS, Routes, router_prompt
from langchain_community.vectorstores import FAISS
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
import os


load_dotenv()

embeddings = get_ollama_embedding("all-minilm:l6-v2", os.getenv("BASE_URL"))
llm = get_ollama("llama3.1", os.getenv("BASE_URL"))
vectorstore = FAISS.load_local(
    "products", embeddings, allow_dangerous_deserialization=True
)

workflow = StateGraph(AgentState)

## Defining the tools
tools = [web_search_tool, get_prebuilt_retriever_tool("""
 data scraped from Micro Labs, a pharmaceutical company in India.
""", vectorstore)]

tool_nodes = {
    "product_detail_agent": ToolNode(tools),
    "recommendation_agent": ToolNode(tools),
    "summarizer_agent": ToolNode(tools),
    "alternatives_agent": ToolNode(tools),
    "general_agent": ToolNode(tools)
}

llm_with_tools = llm.bind_tools(tools)

## Defining the workflow
def router(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    checker = llm.with_structured_output(Routes)
    chain = router_prompt | checker
    decision = chain.invoke(last_message)
    return decision.route

def get_continue(key: str):
    def should_continue(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return f"{key}_tools"
        return END
    return should_continue

def product_model(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(SYSTEM_PROMPTS["product_details"])
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

def recommendation_model(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(SYSTEM_PROMPTS["recommendation"])
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

def summariser_model(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(SYSTEM_PROMPTS["summarizer"])
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

def alternatives_model(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(SYSTEM_PROMPTS["alternatives"])
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}

def call_model(state: AgentState):
    messages = state["messages"]
    system_prompt = SystemMessage(SYSTEM_PROMPTS["general"])
    response = llm_with_tools.invoke([system_prompt] + messages)
    return {"messages": [response]}



## building
config = {"configurable": {"thread_id": "1", "user_id": "1"}}

# workflow.add_node("router", router)
workflow.add_node("product_detail_agent", product_model)
workflow.add_node("recommendation_agent", recommendation_model)
workflow.add_node("summarizer_agent", summariser_model)
workflow.add_node("alternatives_agent", alternatives_model)
workflow.add_node("general_agent", call_model)

workflow.add_conditional_edges(START,router,{
    "product_details": "product_detail_agent",
    "recommendation": "recommendation_agent",
    "summarizer": "summarizer_agent",
    "alternatives": "alternatives_agent",
    "general": "general_agent"
})

for key, node in tool_nodes.items():
    workflow.add_node(f"{key}_tools", node)
    workflow.add_conditional_edges(key, get_continue(key), [f"{key}_tools", END])
    workflow.add_edge(f"{key}_tools", key)

agent = workflow.compile(checkpointer=MemorySaver())


# print_messages_state(agent.get_state(config=config))
msg = str(input("Enter your message: "))
while msg != "exit":
    stream_graph_updates(agent, msg, config=config)
    msg = str(input("Enter your message: "))