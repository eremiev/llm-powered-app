import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### SEQUENTIAL CHAINS

from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain

llm1 = OpenAI(model_name='text-davinci-003', temperature=0.7, max_tokens=1024)
prompt1 = PromptTemplate(
  input_variables=['concept'],
  template = """You are experianced scientist and Python programmer.
  Write a function that implement the concept of {concept}. """
)
chain1 = LLMChain(llm=llm1, prompt=prompt1)

llm2 = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
prompt2 = PromptTemplate(
  input_variables=['function'],
  template = "Given the Python function {function}, describe it as detailed as possible."
)
chain2 = LLMChain(llm=llm2, prompt=prompt2)

overall_chain = SimpleSequentialChain(chains=[chain1,chain2], verbose=True)

# run the overall chain with the first chain input
output = overall_chain.run("softmax")