from typing import Annotated,Sequence,TypedDict
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START,END
from langgraph.prebuilt import ToolNode
from langchain_ollama import ChatOllama
from langchain_core.tools import tool

# Annotated - provides additional context without changing the underlying type

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage],add_messages]

@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    print("Inputs to add:", a, b)
    return a + b

@tool
def subtract(a: int, b: int) -> int:
    """Subtract two numbers."""
    print("Inputs to subtract:", a, b)
    return a - b

@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers."""
    print("Inputs to multiply:", a, b)
    return a * b

tools=[add,subtract,multiply]

model = llm = ChatOllama(
    model="llama3.1:latest", 
    validate_model_on_init = True,
    temperature = 0.8,
    num_predict = 256,
).bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    print("Invoking model...")
    system_prompt= SystemMessage(content=
                                 "you are my AI Assistant, please answer by query best on your ability")
    response = model.invoke([system_prompt] + list(state["messages"]))

    # build new history and convert model response into an AIMessage that may carry tool_calls
    '''
    history = list(state["messages"])
    ai_msg = AIMessage(content=getattr(response, "content", ""))
    '''
    # preserve tool_calls metadata so the ToolNode can detect and run tools
    '''
    if getattr(response, "tool_calls", None):
        setattr(ai_msg, "tool_calls", response.tool_calls)
    history.append(ai_msg)
    '''
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
     # safely check for tool_calls attribute
    # if the model requested a tool, go to the tools node; otherwise end
    if getattr(last_message, "tool_calls", None):
        return "continue"
    return "end"

# Build graph
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)
graph.add_node("tools", ToolNode(tools=tools))
graph.add_edge(START, "our_agent")
graph.add_conditional_edges("our_agent", should_continue,
                             {"end": END, "continue": "tools"})
graph.add_edge("tools", "our_agent")    

agent = graph.compile()

def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        print(message)
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input = {'messages': [HumanMessage(content="Add 5 , 10. Subtract 8 from 20. Multiply 4 and 5. What is the final result?")]}
print_stream(agent.stream(input,stream_mode="values"))
