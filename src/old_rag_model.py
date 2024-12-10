import bs4
import os
from typing import Sequence
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

CHATGPT_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

# Load, chunk and index the contents of the blog to create a retriever.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",
                "https://pyvo.readthedocs.io/en/latest/index.html#",
                "https://pyvo.readthedocs.io/en/latest/registry/index.html",
                "https://pyvo.readthedocs.io/en/latest/discover/index.html",
                "https://pyvo.readthedocs.io/en/latest/io/vosi.html",
                "https://pyvo.readthedocs.io/en/latest/io/uws.html",
                "https://pyvo.readthedocs.io/en/latest/api/pyvo.auth.AuthSession.html#pyvo.auth.AuthSession",
                "https://pyvo.readthedocs.io/en/latest/api/pyvo.auth.AuthURLs.html#pyvo.auth.AuthURLs",
                "https://pyvo.readthedocs.io/en/latest/samp.html",
                "https://pyvo.readthedocs.io/en/latest/api/pyvo.auth.CredentialStore.html#pyvo.auth.CredentialStore",
                "https://pyvo.readthedocs.io/en/latest/mivot/index.html",
                "https://pyvo.readthedocs.io/en/latest/mivot/index.html",
                "https://pyvo.readthedocs.io/en/latest/utils/prototypes.html",
                ),
)
docs = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

vectorstore = QdrantVectorStore.from_documents(
    documents=splits,
    embedding=OpenAIEmbeddings(),
    url=QDRANT_URL,
    prefer_grpc=True,
    api_key=QDRANT_API_KEY,
    collection_name="test_documents",
)
retriever = vectorstore.as_retriever()

# retriever and chain
system_prompt = (
    "You are an assistant for question-answering tasks related to astronomy, astrophysics and any other discipline that derives from this two. "
    "Use the following pieces of retrieved context to answer "
    "any questions. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise unless asked otherwise."
    "\n\n"
    "{context}"
)

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# history
class State(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    context: str
    answer: str


def call_model(state: State):
    response = rag_chain.invoke(state)
    return {
        "chat_history": [
            HumanMessage(state["input"]),
            AIMessage(response["answer"]),
        ],
        "context": response["context"],
        "answer": response["answer"],
    }


workflow = StateGraph(state_schema=State)
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

def get_message(msg_input, config):
  result = app.invoke(
    {"input": msg_input},
    config=config,
  )
  return result["answer"]
