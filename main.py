import streamlit as st
from typing import List
import time
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
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
import os
import pandas as pd

# Page config
st.set_page_config(
    page_title="Pharma Agent Chat 💊",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "agent" not in st.session_state:
    # Initialize the agent
    load_dotenv()
    
    embeddings = get_ollama_embedding("all-minilm:l6-v2", os.getenv("BASE_URL"))
    llm = get_ollama("llama3.1", os.getenv("BASE_URL"))
    vectorstore = FAISS.load_local(
        "products", embeddings, allow_dangerous_deserialization=True
    )

    workflow = StateGraph(AgentState)

    # Tools setup
    tools = [
        web_search_tool,
        get_prebuilt_retriever_tool(
            """data scraped from Micro Labs, a pharmaceutical company in India.""",
            vectorstore
        )
    ]

    tool_nodes = {
        "product_detail_agent": ToolNode(tools),
        "recommendation_agent": ToolNode(tools),
        "summarizer_agent": ToolNode(tools),
        "alternatives_agent": ToolNode(tools),
        "general_agent": ToolNode(tools)
    }

    llm_with_tools = llm.bind_tools(tools)

    # Router function
    def router(state: AgentState):
        messages = state["messages"]
        last_message = messages[-1]
        checker = llm.with_structured_output(Routes)
        chain = router_prompt | checker
        decision = chain.invoke(last_message)
        return decision.route

    # Continue function
    def get_continue(key: str):
        def should_continue(state: AgentState):
            messages = state["messages"]
            last_message = messages[-1]
            if last_message.tool_calls:
                return f"{key}_tools"
            return END
        return should_continue

    # Model functions
    def product_model(state: AgentState):
        messages = state["messages"]
        system_prompt = SystemMessage(content=SYSTEM_PROMPTS["product_details"])
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    def recommendation_model(state: AgentState):
        messages = state["messages"]
        system_prompt = SystemMessage(content=SYSTEM_PROMPTS["recommendation"])
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    def summariser_model(state: AgentState):
        messages = state["messages"]
        system_prompt = SystemMessage(content=SYSTEM_PROMPTS["summarizer"])
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    def alternatives_model(state: AgentState):
        messages = state["messages"]
        system_prompt = SystemMessage(content=SYSTEM_PROMPTS["alternatives"])
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    def call_model(state: AgentState):
        messages = state["messages"]
        system_prompt = SystemMessage(content=SYSTEM_PROMPTS["general"])
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    # Build workflow
    workflow.add_node("product_detail_agent", product_model)
    workflow.add_node("recommendation_agent", recommendation_model)
    workflow.add_node("summarizer_agent", summariser_model)
    workflow.add_node("alternatives_agent", alternatives_model)
    workflow.add_node("general_agent", call_model)

    workflow.add_conditional_edges(
        START,
        router,
        {
            "product_details": "product_detail_agent",
            "recommendation": "recommendation_agent",
            "summarizer": "summarizer_agent",
            "alternatives": "alternatives_agent",
            "general": "general_agent"
        }
    )

    for key, node in tool_nodes.items():
        workflow.add_node(f"{key}_tools", node)
        workflow.add_conditional_edges(
            key,
            get_continue(key),
            [f"{key}_tools", END]
        )
        workflow.add_edge(f"{key}_tools", key)

    st.session_state.agent = workflow.compile(checkpointer=MemorySaver())

def response_generator(msg: str, config: dict):
    input_msg = { "role" : "user", "content": msg }
    agent = st.session_state.agent
    for chunk in agent.stream({"messages": [input_msg]}, stream_mode="messages", config=config):
        # chunk["messages"][-1].pretty_print()
        if isinstance(chunk[0], AIMessage):
            yield chunk[0].content + " "
        elif isinstance(chunk[0], ToolMessage):
            if chunk[0].name=="web_search_tool":
                yield "Searching the web for relevant information... "
                # time.sleep(0.1)
                # yield chunk
            elif chunk[0].name=="doc_retriever":
                yield "Retrieving information from the database..." + " "
                time.sleep(0.1)
                yield "Using the following documents:" + " "
                time.sleep(0.1)
                docs = chunk[0].content.split("\n")
                docs = [doc for doc in docs if doc]
                yield pd.DataFrame(docs, columns=["Documents"]) + " "
        else:
            yield "Processing the information... "
        time.sleep(0.05)


# Streamlit UI
st.title("💊 Pharma Agent Chat")

# Sidebar
with st.sidebar:
    st.header("About")
    st.markdown("""
    This is a LangGraph-powered chatbot that can:
    - Provide product details
    - Make recommendations
    - Summarize information
    - Suggest alternatives
    - Handle general queries
    
    The agent uses tools for web search and product information retrieval.
    """)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Display chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.write(message.content)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to state
    st.session_state.messages.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("user"):
        st.write(prompt)
                
    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": "1", "user_id": "1"}}
    
        # Update agent state with current messages
        agent_state = {"messages": st.session_state.messages}
        response = st.write_stream(response_generator(prompt, config))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
