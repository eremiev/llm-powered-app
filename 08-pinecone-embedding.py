import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### PINECONE EMBEDDING

import pinecone
from langchain.vectorstores import Pinecone

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))


index_name="langchain-pinecone"

if index_name not in pinecone.list_indexes():
  print(f"Creating index {index_name} ...")
  pinecone.create_index(index_name, dimension=1536, metric="cosine", pods=1, pod_type="p1.x2")
  print("Done")
else:
  print(f"Index {index_name} already exists!")


with open("churchill_speech.txt") as f:
  churchill_speech = f.read()

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=100, # ~ a line of text
  chunk_overlap=20,
  length_function=len
)

chunks = text_splitter.create_documents([churchill_speech])
#print(chunks[2].page_content)

embeddings = OpenAIEmbeddings()
#vector =embeddings.embed_query(chunks[0].page_content)
#print(vector)

## Upload the vectors to Pinecone using Langchain
vector_store = Pinecone.from_documents(chunks, embeddings, index_name=index_name)
