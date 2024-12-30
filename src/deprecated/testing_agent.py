from typing import Annotated

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, AIMessage
from typing_extensions import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from custom_tools import get_registry, query_sia

memory = MemorySaver()

class State(TypedDict):
    messages: Annotated[list, add_messages]


graph_builder = StateGraph(State)


tools = [get_registry, query_sia]
llm = ChatOpenAI(model="gpt-4o-mini")
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}


graph_builder.add_node("chatbot", chatbot)
tool_node = ToolNode(tools=[get_registry, query_sia])
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")
graph = graph_builder.compile(checkpointer=memory)# Este es el agente

# print(get_registry.name)
# print(get_registry.description)
# print(get_registry.args)


def get_response(msg_input, config):
    events = graph.stream(
        {"messages": [("user", msg_input)]}, config, stream_mode="values"
    )
    msgs = []
    for event in events:
        if event["messages"][-1].type == 'ai' and len(event["messages"][-1].tool_calls) == 0:
            msgs.append(event["messages"][-1].content)
    
    return msgs[-1]


# config = {"configurable": {"thread_id": "1"}}

# while True:
#     new_input = input(">")
#     if new_input == "quit":
#         break
#     events = graph.stream(
#         {"messages": [("user", new_input)]}, config, stream_mode="values"
#     )
#     for event in events:
#         event["messages"][-1].pretty_print()



# user_input = "Hi there! My name is Vicente."
# # The config is the **second positional argument** to stream() or invoke()!
# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# for event in events:
#     event["messages"][-1].pretty_print()

# user_input = "I need resources from the VO (Virtual Observatory). Specifically, I need resources related to 'white dwarves' of the service 'tap' from the author 'Erik Ferguson'."
# # The config is the **second positional argument** to stream() or invoke()!
# events = graph.stream(
#     {"messages": [("user", user_input)]}, config, stream_mode="values"
# )
# msgs = []
# for event in events:
#     # if event["messages"][-1].type == 'tool' or (event["messages"][-1].type == 'ai' and len(event["messages"][-1].tool_calls) != 0):
#     #     continue
#     if event["messages"][-1].type == 'ai' and len(event["messages"][-1].tool_calls) == 0:
#         msgs.append(event["messages"][-1].content)

# print("Mensajes destinados para el usuario:",end="\n\n")
# for m in msgs:
#     print(type(m))
#     print(m,end="\n\n")