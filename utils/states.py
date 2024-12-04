from typing_extensions import TypedDict
from typing import Annotated, Sequence
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.store.memory import InMemoryStore

class State(TypedDict):
    messages: Annotated[list[str], add_messages]

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

def print_messages_state(state):
    """Print the state information in a readable format"""
    print("\n" + "="*80 + "\n")
    print("CONVERSATION HISTORY:")
    print("-"*40)
    
    # Print messages
    for msg in state.values['messages']:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        print(f"\n{role}:")
        print(f"{msg.content}\n")
        
        # Print metadata if it exists
        if hasattr(msg, 'usage_metadata'):
            print("Tokens:")
            print(f"  Input: {msg.usage_metadata['input_tokens']}")
            print(f"  Output: {msg.usage_metadata['output_tokens']}")
            print(f"  Total: {msg.usage_metadata['total_tokens']}")
            
        if hasattr(msg, 'response_metadata') and isinstance(msg, AIMessage):
            duration = msg.response_metadata.get('total_duration', 0) / 1_000_000  # Convert to milliseconds
            print(F"tool_calls: {msg.tool_calls}")
            print(f"Response time: {duration:.2f}ms")
    
    print("\n" + "="*80)