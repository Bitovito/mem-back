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
from datetime import datetime
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from typing import Optional
from langgraph.prebuilt import ToolNode
from .data_types import VoToolResponse
from .custom_tools import get_registry, query_sia, query_ssa, query_scs, query_sla, retriever_tool


class AgentState(TypedDict):
    """The state of the agent."""
    last_tool_called: Optional[str] = None  # Track the last tool called
    tool_data: Optional[Any] = None # Store structured output here if needed
    messages: Annotated[Sequence[BaseMessage], add_messages]


model = ChatOpenAI(model="gpt-4o-mini")

memory = MemorySaver()

tools = [get_registry, query_sia, query_ssa, query_scs, query_sla, retriever_tool]

model_with_tools = model.bind_tools(tools, tool_choice="auto")

tools_by_name = {}
for tool in tools:
    tools_by_name[tool.name] = tool

def grade_documents(state) -> Literal["generate", "rewrite"]:
    print("---CHECK RELEVANCE---")
    
    class grade(BaseModel):
        """Binary score for relevance check."""
        binary_score: str = Field(description="Relevance score 'yes' or 'no'")

    # LLM
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)

    # LLM with structured output
    llm_with_structured_output = model.with_structured_output(grade)

    # Prompt
    prompt = PromptTemplate(
        template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
        Here is the retrieved document: \n\n {context} \n\n
        Here is the user question: {question} \n
        If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
        Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question.""",
        input_variables=["context", "question"],
    )

    # Chain
    chain = prompt | llm_with_structured_output

    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    scored_result = chain.invoke({"question": question, "context": docs})
    score = scored_result.binary_score

    if score == "yes":
        print("---DECISION: DOCS RELEVANT---")
        return "generate"

    else:
        print("---DECISION: DOCS NOT RELEVANT---")
        print(score)
        return "rewrite"

def rewrite(state):
    print("---TRANSFORM QUERY---")
    question = state["messages"][0].content
    msg = [
        HumanMessage(
            content=f""" \n 
    Look at the input and try to reason about the underlying semantic intent / meaning. \n 
    Here is the initial question:
    \n ------- \n
    {question} 
    \n ------- \n
    Formulate an improved question: """,
        )
    ]
    
    model = ChatOpenAI(temperature=0, model="gpt-4-0125-preview", streaming=True)
    response = model.invoke(msg)
    return {"messages": [response]}

def generate(state):
    print("---GENERATE---")
    messages = state["messages"]
    question = messages[0].content
    docs = messages[-1].content
    # Prompt
    prompt = hub.pull("rlm/rag-prompt")
    # LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, streaming=True)
    # Post-processing
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    # Chain
    rag_chain = prompt | llm | StrOutputParser()
    # Run
    response = rag_chain.invoke({"context": docs, "question": question})
    return {"messages": [response]}


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
                name = tool_call["name"],
                tool_call_id = tool_call["id"]
            )
        )
        if isinstance(tool_result, VoToolResponse):
            last_tool_results = tool_result.data_response

    return {"messages": outputs,
            "last_tool_called": outputs[-1].name,
            "tool_data": last_tool_results}

retrieve = ToolNode([retriever_tool])

# Agent Node
def call_model(state: AgentState, config: RunnableConfig):
    print("Llamada a LLM...")###
    current_date = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

    system_prompt = SystemMessage(
        """The current date is {0}. You are an assistant to astronomers and people interested in astronomy. Your speciality is aswering questions related to the Virtual Observatory (VO), by 
        either answering questions about it (using the retriever tool) or returning data and information from it's services using the VO tools (this tools are anything but the retriever tool). 
        When a VO tool is executed, it will respond to you with a success message describing the state of success or an error message that states why it failed. When successful, the VO tool will 
        create data and store it in the state of this application, which you do not have access to. DO NOT CREATE YOUR OWN ANSWER if you called a VO tool, just limit yourself to inform the user 
        of the success of the query and any description the VO tool gave you. 
        
        If the user asks for information related to the Virtual Observatory, IVOA or any astronomical and cientific questions, remember to evauale if using the retriever could aid you in providing a more complete answer.
        If the user asks for data or information realted to an astronomical object, not specifying what kind, use get_registry. 
        If they ask for images, graphic data or fits data, use the query_sia tool. Before executing this tool, make sure you understand what type of graphic data the user is asking for.
        If they ask for you to 'show them' 'an' image or 'one' image, use get_img. 
        If they ask for you to perform a conesearch, use the tool query_scs. 
        If they ask you for 'spectral data' of an object, use the tool query ssa. 
        If they ask for 'spectral data' in a wavelength range, use query_sla. 

        Do not use a url unless asked to. Finally, it is of incredible importance that you use the appropriate tool, don't forget this.
        """.format(current_date)
    )
    response = model_with_tools.invoke([system_prompt] + state["messages"], config)

    print(f"Respuesta completa del modelo:\n{response}")###
    #####
    # if hasattr(response, "tool_calls") and len(response.tool_calls) != 0:
    #     last_tool_called = response.tool_calls[-1]['name']
    #     return {"messages": [response], "last_tool_called": last_tool_called}
    #####
    return {"messages": [response]}


# Conditional edge; determines whether to continue or not
def route_to(state: AgentState) -> Literal["tools", "retrieve", "__end__"]:
    print("Calculando siguiente nodo...")###

    messages = state["messages"]
    last_message = messages[-1]

    print(f"Numero de llamadas a tools: {len(last_message.tool_calls)}\nÚltima tool llamada: " + state["last_tool_called"])###
    
    # if len(last_message.tool_calls) == 1 and state["last_tool_called"] in structured_classes:
    #     return "respond"
    if len(last_message.tool_calls) > 0:
        if last_message.tool_calls[-1]["name"] == "retrieve_scientific_documents":
            return "retrieve"
        return "tools"
    
    else:
        return "__end__"


workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_node("retrieve", retrieve)
workflow.add_node("rewrite", rewrite)
workflow.add_node("generate", generate)

workflow.set_entry_point("agent")

workflow.add_conditional_edges(
    "agent",
    route_to,
    {
        "tools": "tools",
        "retrieve": "retrieve",
        END: END
    }
)
# Edges taken after the `retrieve` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate", END)
workflow.add_edge("rewrite", "agent")

workflow.add_edge("tools", "agent")

graph = workflow.compile(checkpointer=memory)# Este es el agente final
# graph.get_graph().draw_mermaid_png(output_file_path="rag_artifact_graph.png")

######################################################################

async def get_response(msg_input, config):
    print(f"Mensaje usuario:\n{msg_input}")###

    final_state = await graph.ainvoke({
        "messages": [("human", msg_input)],
        "last_tool_called": "None"
        }, config)    
    
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

