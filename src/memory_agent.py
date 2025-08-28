from typing import TypedDict,Union,List
from langchain_core.messages import HumanMessage,AIMessage
from langgraph.graph import StateGraph, START,END
from langchain_ollama import ChatOllama

llm = ChatOllama(
    model="llama3.1:latest", 
    validate_model_on_init = True,
    temperature = 0.8,
    num_predict = 256,
)

class AgentState(TypedDict):
    messages: List[Union[HumanMessage,AIMessage]]

def process_message(state: AgentState) -> AgentState:
    """Process the incoming message and generate a response using Llama3."""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content))
    print(f"\n AI: {response.content}")
    return state

# Build graph
graph = StateGraph(AgentState)
graph.add_node("process", process_message)  
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

if __name__ == "__main__":
    conversation_history: List[Union[HumanMessage,AIMessage]] = []
    user_input = input("Enter your message (or type 'exit' to quit): ")    
    #result = agent.invoke(initial_state)

    while user_input.lower() != "exit":

        conversation_history.append(HumanMessage(content=user_input))
        initial_state = {"messages": conversation_history}
        result = agent.invoke(initial_state)
        conversation_history = result["messages"]

        user_input = input("Enter your message (or type 'exit' to quit): ")
    
    print("Saving the conversation history...")
    with open("conversation_history.txt", "w") as f:
        for msg in conversation_history:
            role = "User" if isinstance(msg, HumanMessage) else "AI"
            f.write(f"{role}: {msg.content}\n")
    print("Conversation history saved to conversation_history.txt")

