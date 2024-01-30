from langchain_community.llms import openai
from langchain_openai import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
#from langchain.tools import DuckDuckGoSearchTool
from subprocess import Popen,PIPE
from langchain.agents import initialize_agent
import os
from dotenv import load_dotenv
import json



from langchain_community.tools import DuckDuckGoSearchRun
from langchain.agents import Tool
from langchain.tools import BaseTool

class ExecuteCommand(BaseTool):
    name= "execute_command"
    description = "execute commands in the VM"

    def _run(self, command: str):
        cmd = command.split()
        if cmd[0]=='sudo':
            cmd.insert(1,'-S')
        print(cmd)
        """"
        p = Popen(cmd ,stdin = PIPE, stdout = PIPE, universal_newlines = True)
        p.stdin.write('test1!')
        res,_ = p.communicate()
    
        p.terminate()

        return res
        """
        return json.dumps(cmd)       

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")

execute_command = ExecuteCommand()


def main():
    load_dotenv()

    search = DuckDuckGoSearchRun()
    llm = ChatOpenAI(temperature=0, model_name = "gpt-3.5-turbo-1106", api_key=os.getenv('OPEN_API_KEY'))
    tools = [
        Tool(
            name="search",
            func = search.run,
            description = "useful for when you nee to answer questions about current events."
        ),execute_command
    ]    
    
    # conversational agent memory
    memory = ConversationBufferWindowMemory(
                        memory_key='chat_history',
                        k=3,
                        return_messages=True
                        )

    #print(search.run("Quelle est la capitale fédérale du Canada?"))

    # create our agent
    conversational_agent = initialize_agent(
            agent='chat-conversational-react-description',
            tools=tools,
            llm=llm,
            verbose=True,
            max_iterations=3,
            early_stopping_method='generate',
            memory=memory
        )
    conversational_agent("Give a command to run the updating apt.")

if __name__ == "__main__":
    main()