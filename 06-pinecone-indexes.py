import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### PINECONE INDEXES
# An index is the highest level organizational unit of vector data in Pinecone.
# It accepts and stores vectors serves queries over the vectors it contains and does other vector operations

import pinecone

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment=os.environ.get("PINECONE_ENV"))


index_name="langchain-pinecone"

if index_name not in pinecone.list_indexes():
  print(f"Creating index {index_name} ...")
  pinecone.create_index(index_name, dimension=1536, metric="cosine", pods=1, pod_type="p1.x2")
  print("Done")
else:
  print(f"Index {index_name} already exists!")


#pinecone.describe_index(index_name)
#pinecone.delete_index(index_name)

#To do any operation with Index you must first select it
index = pinecone.Index(index_name)
#print(index.describe_index_stats())



### PINECONE NAMESPACES