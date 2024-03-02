from dotenv import load_dotenv
from langchain.agents import Tool, AgentOutputParser, initialize_agent, AgentType
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI, ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.tools import BaseTool
from subprocess import Popen,PIPE
from typing import List
from langchain.schema import AgentAction, AgentFinish
import re, os, json
import langchain
from langchain.memory import ConversationBufferWindowMemory

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA


from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

load_dotenv()
class Transfrom_to_Bash(BaseTool):
    name : str = "Transform to Bash",
    description= "translate the text to bash command to be executed in the VM"

    def _run(self, command: str):
        return command

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")

def AskvectorDB(question: str):
    embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002')
    index_name = 'first-index'

    vector_store = Pinecone.from_existing_index(index_name,embeddings)   
    llm2 = ChatOpenAI(model_name='gpt-3.5-turbo-1106', temperature=0.0)
    # retrieval qa chain
    qa = RetrievalQA.from_chain_type(
        llm=llm2,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    response = qa.run(question)
    return response

llm = OpenAI(temperature=0.0)

class Data1(BaseModel):
    subject: str = Field(description="The subject of the prompt")
    steps: List[str] = Field(description="The list of multiples actions to be performed")
output_parser1 = PydanticOutputParser(pydantic_object=Data1)
format_instruction1 = output_parser1.get_format_instructions()

agent_tools1 = [ 
    Tool.from_function(
        name ="askvector_db", 
        func = AskvectorDB,
        description = "this tool is to ask the vector db."
    ),]
prompt_template1 = PromptTemplate(template="""Get the response of {subject} in the tools. {format_instruction} """, tools=agent_tools1, partial_variables = {"format_instruction": format_instruction1} ,input_variables=["subject"])


class Data2(BaseModel):    
    commands: List[str] = Field(description="The list of commands to be executed in a VM")
output_parser2 = PydanticOutputParser(pydantic_object=Data2)
format_instruction2 = output_parser2.get_format_instructions()

agent_tools2= [ 
    Tool(
        name ="Transform_to_Bash", 
        func = Transfrom_to_Bash.run,
        description = "this tool is to transform the text to bash commands."
    ),]

memory = ConversationBufferWindowMemory()
prompt_template2 = PromptTemplate(template="""for given each line for the steps {steps}, translate it to bash command to be executed in a VM.
                                  
                            Previous conversation history:
                            {history}
                            {format_instruction} """, tools=agent_tools2, partial_variables = {"format_instruction": format_instruction2} ,input_variables=["steps", "history"])



llm_chain1 = LLMChain(llm=llm, prompt=prompt_template1, output_key="steps")
llm_chain2 = LLMChain(llm=llm, prompt=prompt_template2, output_key="commands")
two_chains = SequentialChain(chains = [llm_chain1, llm_chain2],
                         input_variables=["subject", "history"],
                         output_variables=["commands"],
                         verbose=True)

"""

agent = initialize_agent(tools=agent_tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

res = agent.run(prompt_template.format_prompt(subject="how to protect data on your devices?"))
 Previous conversation history:
                                {history} 

"""
while True:
    input_text = input("Enter the question: ")
    if input_text == "":
        break
    res = two_chains({"subject":input_text, "history":memory})
    print(res)
