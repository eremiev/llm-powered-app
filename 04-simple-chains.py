import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### SIMPLE CHAINS

from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

template = """You are an experianced virologist.
Write a few sentences about the following {virus} in {language}"""

prompt = PromptTemplate(
  input_variables=['virus', 'language'],
  template = template
)

llm = OpenAI(model_name='text-davinci-003', temperature=0.7)
output = llm(prompt.format(virus='ebola',language='bulgarian'))

print(output)
