import os
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(), override=True)

### CHAT MODELS GPT-3.5 Turbo / GPT-4

# User (HumanMessage) - what we ask the assistant
# Assistant (AIMessage) - help store prior messages
# System (SystemMessage) - helps set the behavior of the system

from langchain.schema import (
  AIMessage,
  HumanMessage,
  SystemMessage
)
from langchain.chat_models import ChatOpenAI


chat = ChatOpenAI(model_name='gpt-3.5-turbo', temperature=0.5, max_tokens=1024)
messages = [
  SystemMessage( content = 'You are a physicist and respond only in Bulgarian'),
  HumanMessage( content = 'explain quantum mechanics in one sentence')
]

output = chat(messages)
print(output.content)
