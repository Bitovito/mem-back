from dotenv import load_dotenv
import os

load_dotenv()

# Access the environment variables from the .env file
open_api_key = os.environ.get('OPENAI_API_KEY')
lang_api_key = os.environ.get('LANGCHAIN_API_KEY')
lang_tracing = os.environ.get('LANGCHAIN_TRACING_V2')
# print(open_api_key, lang_api_key, lang_tracing)

import json
import inspect
from typing import (
    Annotated,
    Any,
    Literal,
    Sequence,
    TypedDict,
)
from pydantic import BaseModel
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from typing import Optional
from .data_types import RegistryResponseParser, ImageResponseParser, VoToolResponse
from .custom_tools import get_registry, query_sia, query_ssa, query_scs, query_sla


class AgentState(TypedDict):
    """The state of the agent."""
    last_tool_called: Optional[str] = None  # Track the last tool called
    tool_data: Optional[Any] = None # Store structured output here if needed
    messages: Annotated[Sequence[BaseMessage], add_messages]


model = ChatOpenAI(model="gpt-4o-mini")

memory = MemorySaver()

structured_classes = []

# tools = [get_registry, RegistryResponseParser, query_sia, ImageResponseParser, query_ssa, query_scs, query_sla]
tools = [get_registry, query_sia, query_ssa, query_scs, query_sla]

model_with_tools = model.bind_tools(tools, tool_choice="auto")

tools_by_name = {}
for tool in tools:
    #####
    if inspect.isclass(tool):
        tools_by_name[tool.__name__] = tool
        structured_classes.append(tool.__name__)
    #####    
    else:
        tools_by_name[tool.name] = tool

# tools_by_name = {tool.name: tool for tool in tools}


# Tool Node
def tool_node(state: AgentState):
    print("Ejecución de herramientas...")###

    outputs = []
    last_tool_results = None
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])        
        outputs.append(
            ToolMessage(
                content = json.dumps(tool_result.semantic_response) if isinstance(tool_result, VoToolResponse) else json.dumps(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
        )
        if isinstance(tool_result, VoToolResponse):
            last_tool_results = tool_result.data_response

    return {"messages": outputs,
            "last_tool_called": outputs[-1].name,
            "tool_data": last_tool_results}


# Agent Node
def call_model(state: AgentState, config: RunnableConfig):
    print("Llamada a LLM...")###

    system_prompt = SystemMessage(
        """You are an assistant to astronomers and people interested in astronomy. Your speciality is aswering questions related to the virtual Observatory 
        and return data from it using the tools you will be provided with. When a tool of this is executed, it will respond to you with a success message 
        describing the state of success or an error message that states why it failed. When successful, the tool will create data and store it in the state 
        of this application, which is you do not have access to. DO NOT CREATE YOUR OWN ANSWER if you called a tool, just limit yourself to inform the user 
        of the success of the query and any desription the tool gave you. 
          """
        #   If you recieve an answer from one of the tools you use, it is IMPERATIVE that the nex tool you call is a parser for 
        #   that tools output. The user does not need to know of this"""
    )
    response = model_with_tools.invoke([system_prompt] + state["messages"], config)

    print(f"Respuesta completa del modelo:\n{response}")###
    #####
    if hasattr(response, "tool_calls") and len(response.tool_calls) != 0:
        last_tool_called = response.tool_calls[-1]['name']
        return {"messages": [response], "last_tool_called": last_tool_called}
    #####
    return {"messages": [response]}


# async def respond(state: AgentState):
#     print(f"Estructurando respuesta...")###

#     match state["last_tool_called"]:
#         case "RegistryResponseParser":
#             response = RegistryResponseParser(**state["messages"][-1].tool_calls[0]["args"])
#         case "ImageResponseParser":
#             print(f"A ver los args de la imagen: {state['messages'][-1].tool_calls[0]['args']}")
#             response = ImageResponseParser(**state["messages"][-1].tool_calls[0]["args"])
#         case _:
#             print("Esta función no requiere reestructuramiento: " + state["last_tool_called"] )### Debería distinguir entre mensajes simples y de herramienta.
#             response = state["messages"][-1]

#     reqToolMsg = ToolMessage(
#         content=response,
#         name=state["last_tool_called"],
#         tool_call_id=state["messages"][-1].tool_calls[0]["id"],
#     )

#     print("Final structured response:", response)###
#     return {"messages": [reqToolMsg],
#             "final_response": response}


# Conditional edge; determines whether to continue or not
def route_to(state: AgentState) -> Literal["tools", "__end__"]:
    print("Calculando siguiente nodo...")###

    messages = state["messages"]
    last_message = messages[-1]

    print(f"Numero de llamadas a tools: {len(last_message.tool_calls)}\nÚltima tool llamada: " + state["last_tool_called"])###
    
    # if len(last_message.tool_calls) == 1 and state["last_tool_called"] in structured_classes:
    #     return "respond"
    if len(last_message.tool_calls) > 0:
        return "tools"
    else:
        return "__end__"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
# workflow.add_node("respond", respond)
workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    route_to,
    {
        "tools": "tools",
        # "respond": "respond",
        END: END
    }
)


workflow.add_edge("tools", "agent")
# workflow.add_edge("respond", END)

graph = workflow.compile(checkpointer=memory)# Este es el agente final

######################################################################

async def get_response(msg_input, config):
    print(f"Mensaje usuario:\n{msg_input}")###

    final_state = await graph.ainvoke({
        "messages": [("human", msg_input)],
        "last_tool_called": "None"
        }, config)
    
    # response = {
    #     "AIMessage":final_state.get("final_response"),
    #     "ToolMessage":{final_state.get("messages")[-2]}
    # }
    # return response
    
    
    if final_state.get("last_tool_called") not in {None, "None", "none"}:### check condition for artifact
        return {
            "last_tool_called":final_state.get("last_tool_called"),
            "last_message":final_state.get("messages")[-1].content, 
            # "final_response":final_state.get("final_response")
            "tool_data": final_state.get("tool_data")}
    
    return {"last_message":final_state.get("messages")[-1].content}
        
    # events = graph.stream(
    #     {"messages": [("user", msg_input)]}, config, stream_mode="values"
    # )
    # msgs = []
    # for event in events:
    #     if event["messages"][-1].type == 'ai' and len(event["messages"][-1].tool_calls) == 0:
    #         msgs.append(event["messages"][-1].content)
    #         event["messages"][-1].pretty_print()
    
    # return msgs[-1]


# # Interactuar con el agente
# # I need resources of 'white dwarf' in the 'UV' waveband
# config = {"configurable": {"thread_id": "1"}}

# while True:
#     new_input = input("> ")
#     if new_input == "quit":
#         break
#     for event in graph.stream(
#         {"messages": [("user", new_input)],"last_tool_called": "None"}, config, stream_mode="updates"):
#         for node, values in event.items():
#             print(f"Receiving update from node: '{node}'")
#             if node == 'tools':
#                 print(values["messages"][-1].name)
#                 continue
#             print(values)
#             print("\n\n")

# from IPython.display import Image, display
# from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# display(
#     Image(
#         graph.get_graph().draw_mermaid_png(
#             draw_method=MermaidDrawMethod.API,
#             output_file_path="./new_artifact_graph.png"
#         )
#     )
# )