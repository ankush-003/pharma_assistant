# Utils

## Checkpointer

Checkpoint is a snapshot of the graph state saved at each super-step and is represented by a `StateSnapshot` object with the following key properties:

- **config**: Config associated with this checkpoint.
- **metadata**: Metadata associated with this checkpoint.
- **values**: Values of the state channels at this point in time.
- **next**: A tuple of the node names to execute next in the graph.
- **tasks**: A tuple of `PregelTask` objects that contain information about the next tasks to be executed. If the step was previously attempted, it will include error information. If a graph was interrupted dynamically from within a node, `tasks` will contain additional data associated with interrupts.

```python
checkpointer = MemorySaver()
graph = workflow.compile(checkpointer=checkpointer)
```

## MemoryStore

A state schema specifies a set of keys that are populated as a graph is executed. As discussed above, state can be written by a checkpointer to a thread at each graph step, enabling state persistence.

But, what if we want to retain some information across threads? Consider the case of a chatbot where we want to retain specific information about the user across all chat conversations (e.g., threads) with that user!

With checkpointers alone, we cannot share information across threads. This motivates the need for the Store interface. As an illustration, we can define an `InMemoryStore` to store information about a user across threads. We simply compile our graph with a checkpointer, as before, and use our new `in_memory_store`.

Each memory type is a Python class with certain attributes. We can access it as a dictionary by converting via `.dict` as above. The attributes it has are:

- **value**: The value (itself a dictionary) of this memory.
- **key**: The UUID for this memory in this namespace.
- **namespace**: A list of strings, the namespace of this memory type.
- **created_at**: Timestamp for when this memory was created.
- **updated_at**: Timestamp for when this memory was updated.

## Tool Calling
ToolNode operates on graph state with a list of messages. It expects the last message in the list to be an AIMessage with tool_calls parameter.

```python
message_with_single_tool_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"location": "sf"},
            "id": "tool_call_id",
            "type": "tool_call",
        }
    ],
)

tool_node.invoke({"messages": [message_with_single_tool_call]})
```

output:
```
{'messages': [ToolMessage(content="It's 60 degrees and foggy.", name='get_weather', tool_call_id='tool_call_id')]}
```
llm gives us the arguments to be passed to the tool in the `tool_call` object. We can use this to invoke the tool and get the result. We can then return this result as a `ToolMessage` object.
```python
# Define our tool node
def tool_node(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}
```