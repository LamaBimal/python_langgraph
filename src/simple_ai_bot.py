from typing import TypedDict,List
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

class AgentState(TypedDict):
    message: List[HumanMessage]

llm = ChatOllama(
    model="llama3.1:latest", 
    validate_model_on_init = True,
    temperature = 0.8,
    num_predict = 256,
)

def process_message(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print(f"\n Response from Llama3: {response.content}")
    return state

# Build graph
graph = StateGraph(AgentState)
graph.add_node("process", process_message)
graph.add_edge(START, "process")
graph.add_edge("process", END)

if __name__ == "__main__":
    
    app = graph.compile()
    user_input = input("Enter your message: ")
    initial_state = {"message": [HumanMessage(content=user_input)]}
    result = app.invoke(initial_state)

    while user_input.lower() != "exit":
        user_input = input("Enter your message (or type 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        initial_state = {"message": [HumanMessage(content=user_input)]}
        result = app.invoke(initial_state)