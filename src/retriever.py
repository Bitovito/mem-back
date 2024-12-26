import os
import json
from uuid import uuid4

from langchain_community.document_loaders import WebBaseLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams


qdrant_url = os.environ.get('QDRANT_URL')
qdrant_api_key = os.environ.get('QDRANT_API_KEY')
open_api_key = os.environ.get('OPENAI_API_KEY')
collection_name = os.environ.get('QDRANT_COLLECTION_NAME')

urls = []
with open('src/documents.json', 'r+') as file:
  data = json.load(file)
  for doc in data:
    if not doc["is_loaded"]:
      urls.append(doc["url"])
      doc["is_loaded"] = True

  json_data = json.dumps(data, indent=2)
  file.seek(0)
  file.write(json_data)
  file.truncate()
  file.close()

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=open_api_key)

qdrant_client = QdrantClient(
  url = qdrant_url,
  api_key = qdrant_api_key)

# Check if collection exists, if not, create it.
if qdrant_client.collection_exists(collection_name):
  vectorstore = QdrantVectorStore.from_existing_collection(
    embedding = embeddings,
    collection_name = collection_name,
    url = qdrant_url,
    api_key = qdrant_api_key
  )
else:
  vectorstore = QdrantVectorStore.from_documents(
    doc_splits,
    embeddings,
    url = qdrant_url,
    prefer_grpc = True,
    api_key = qdrant_api_key,
    collection_name = collection_name,
)

if len(doc_splits) > 0:
  uuids = [str(uuid4()) for _ in range(len(doc_splits))]
  vectorstore.add_documents(documents=doc_splits, ids=uuids)


retriever = vectorstore.as_retriever()