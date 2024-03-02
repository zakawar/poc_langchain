from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.llms import openai
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains import LLMChain
from langchain.tools import BaseTool
from subprocess import Popen,PIPE
from typing import List, Union
from langchain.schema import AgentAction, AgentFinish
import re, os, json
import langchain
from langchain.memory import ConversationBufferWindowMemory

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains import RetrievalQA
from langchain.output_parsers import CommaSeparatedListOutputParser

class AskvectorDB(BaseTool):
    name= "askvector_db"
    description = "ask the vector db"

    def _run(self, question: str):

        embeddings = OpenAIEmbeddings(model = 'text-embedding-ada-002', openai_api_key=os.getenv('OPENAI_API_KEY'))
        index_name = 'first-index'

        vector_store = Pinecone.from_existing_index(index_name,embeddings)   
        llm2 = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'), model_name='gpt-3.5-turbo-1106', temperature=0.0)
        # retrieval qa chain
        qa = RetrievalQA.from_chain_type(
            llm=llm2,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        response = qa.run(question)
        return response
                                              
    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")

def main():
    load_dotenv()
    
    askvector_db = AskvectorDB()

    llm = OpenAI(temperature=0.0, api_key=os.getenv('OPEN_API_KEY'))
    tools = [ 
        Tool(
            name ="askvector_db", 
            func = askvector_db._run,
            description = "ask the vector db"
        ),         
    ]  

    output_parser = CommaSeparatedListOutputParser()
    format_instruction = output_parser.get_format_instructions()
    question = ""
    template = PromptTemplate.from_template(f"Start looking for the answer by using the tools available first. You have to answer the question {question} with a detailed plan by putting on points.You have access to the following tools: {tools} with {format_instruction}")
    
    llm_chain = LLMChain(llm=llm, prompt=template, output_parser=output_parser)
    

    #res = run(agent,tools,"How to protect your business?", memory)
    res = llm_chain.run({"question":"How to protect your business?", })
    print(res)


def run(agent, tools, request,memory, verbose=True):
    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent,
                                                    tools=tools,
                                                    verbose=verbose,
                                                    memory=memory
                                                   )
    return agent_executor.invoke(request) 

if __name__ == "__main__":
    main()

