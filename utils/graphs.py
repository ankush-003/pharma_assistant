from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.config import RunnableConfig
from langgraph.prebuilt import ToolNode
from .prompts import MemoryKV, memory_update_prompt, grade_document_prompt, Grade
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage
from .llms import get_ollama
import uuid
import os
from typing import Literal

from .states import State

## Prebuilt Graphs
def get_react_agent(graph: StateGraph, llm, tools):
    """
    Get the tool agent graph
    """
    tool_node = ToolNode(tools)
    llm_with_tools = llm.bind_tools(tools)

    def should_continue(state: State):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: State):
        messages = state["messages"]
        system_prompt = SystemMessage(
            "You are a helpful AI assistant, please respond to the users query to the best of your ability!"
        )
        response = llm_with_tools.invoke([system_prompt] + messages)
        return {"messages": [response]}

    workflow = StateGraph(State)

    # Define the two nodes we will cycle between
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue, ["tools", END])
    workflow.add_edge("tools", "agent")

    return workflow

## Nodes

# This node updates the memory with the key-value pair
def update_memory(state, config: RunnableConfig, *, store):
    """
    Update the memory with the key-value pair
    """
    # Get the user id from the config
    user_id = config["configurable"]["user_id"]

    # Namespace the memory
    namespace = (user_id, "memories")

    # Analyzing
    conversation = state["messages"]

    # Extract the key-value pair
    chain = memory_update_prompt | get_ollama("qwen2.5", os.getenv("BASE_URL")).with_structured_output(MemoryKV)
    memory_dict = chain.invoke({"conversation": conversation})
    memory = {}
    for key, value in memory_dict.value.items():
        memory[key] = value

    # Create a new memory ID
    memory_id = str(uuid.uuid4())

    # We create a new memory
    store.put(namespace, memory_id, {"memory": memory})

# Document grade node
def grade_document(state) -> Literal["generate", "rewrite"]:
    llm = get_ollama("qwen2.5", os.getenv("BASE_URL")).with_structured_output(Grade)

    chain = grade_document_prompt | llm
    messages = state["messages"]
    last_message = messages[-1]

    question = messages[0].content
    docs = last_message.content

    scored_result = chain.invoke({"question": question, "context": docs})

    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"


## Utility functions
def stream_graph_updates(graph: StateGraph, user_input: str, config: RunnableConfig):
    input_message = { "role" : "user", "content": user_input }
    for chunk in graph.stream({"messages": [input_message]}, stream_mode="values", config=config):
        chunk["messages"][-1].pretty_print()