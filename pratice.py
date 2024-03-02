from langchain_openai import OpenAI, ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()
'''
llm = OpenAI(temperature=0.0, api_key=os.getenv('OPEN_API_KEY'))    


prompt = PromptTemplate(template=
                        """List bash commands about {subject}, replace the specific parameter to be inserted by {hue} """, 
                        input_variables=["subject", "hue"])
                        

prompt = PromptTemplate.from_template("""List bash commands about {subject}, 
                                      replace the specific parameter to be inserted by {hue} """)
request = prompt.format(subject="how to create a file in linux", hue="test")

res = llm(request)

print(res)
'''

llm2 = ChatOpenAI(temperature=0.0, api_key=os.getenv('OPEN_API_KEY'))
messages = [
    ("system", "You are a expert in linux machine"),
    "human","List bash commands about {subject}, replace the specific parameter to be inserted by {hue}"
]
prompt = ChatPromptTemplate.from_messages(messages)

message_prompt = prompt.format_messages(subject="how to create a file in linux", hue="test")

res = llm2(message_prompt)

print(res.content)

