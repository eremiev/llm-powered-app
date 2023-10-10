import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### ASK QUESTION FROM PRELOAD INDEX

import pinecone
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

index_name="langchain-pinecone"

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))

embeddings = OpenAIEmbeddings()

# if you already have an index, you can load it like this
docsearch = Pinecone.from_existing_index(index_name, embeddings)


### ASK QUESTIONS
query="Where should we fight?"

docs = docsearch.similarity_search(query)

for r in docs:
  print(r.page_content)
  print("-" * 50)


from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI


llm = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=1)

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# chain_type="stuff" use all the text  from the docs in the prompt

chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

answer=chain.run(query)

print(f"Answer: {answer}")