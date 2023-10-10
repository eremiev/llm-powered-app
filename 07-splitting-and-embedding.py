import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### Splitting and Embedding

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

with open("churchill_speech.txt") as f:
  churchill_speech = f.read()

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=100, # ~ a line of text
  chunk_overlap=20,
  length_function=len
)

chunks = text_splitter.create_documents([churchill_speech])
#print(chunks[2].page_content)


embedding = OpenAIEmbeddings()
vector =embedding.embed_query(chunks[0].page_content)
print(vector)