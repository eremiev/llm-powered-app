import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### TEXT DAVINCHI 003

from langchain.llms import OpenAI

llm = OpenAI(model_name='text-davinci-003', temperature=0.7, max_tokens=512)
prompt = 'write original tagline for burger restaurant'
#output = llm(prompt)
#print(output)

### Get the tokens in this request
#print(llm.get_num_tokens(prompt))

### Provide more than one prompt / generate more than one answer
output = llm.generate([prompt]* 3)
for o in output.generations:
  print(o[0].text)