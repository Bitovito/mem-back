import json
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

urls = []
with open('src/documents.json', 'r+') as file:
  data = json.load(file)
  for doc in data:
    if not doc["is_loaded"]:
      urls.append(doc["url"])
      # doc["is_loaded"] = True

  # json_data = json.dumps(data, indent=2)
  # file.seek(0)
  # file.write(json_data)
  # file.truncate()
  # file.close()

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)

c=1
for split in doc_splits:

  print(f"////////////////////////////Split {c}//////////////////////////////////////")
  print(split)
  print(f"///////////////////////////////////////////////////////////////////////////")
  c+=1