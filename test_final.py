from dotenv import load_dotenv
from langchain_openai import OpenAI
from langchain.agents import Tool, initialize_agent
from subprocess import Popen,PIPE
from langchain.tools import BaseTool
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

class Transfrom_to_Bash(BaseTool):
    name : str = "Transform to Bash",
    description= "translate the text to bash command to be executed in the VM"

    def _run(self, command: str):
        return command

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")

class ExecuteCommand(BaseTool):
    name= "execute_command"
    description = "execute commands in the VM"

    def _run(self, command: str):
        cmd = command.split()
        if cmd[0]=='sudo':
            cmd.insert(1,'-S')
        #print(cmd)
        
        p = Popen(cmd ,stdin = PIPE, stdout = PIPE, universal_newlines = True)
        p.stdin.write('test1!')
        res,_ = p.communicate()
    
        p.terminate()

        return res
        
        #return json.dumps(cmd)       

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")


search = DuckDuckGoSearchRun()
transform_to_bash = Transfrom_to_Bash()
execute_command = ExecuteCommand()

tools = [ 
        Tool(
            name ="search", 
            func = search.run,
            description = "search the web for information."
        ),   
        Tool(
            name ="transform_to_bash", 
            func = transform_to_bash._run,
            description = "translate the text to bash command to be executed in the VM"
        ),       
        Tool(
            name ="execute_command", 
            func = execute_command._run,
            description = "execute commands in the VM"
        ),
    ]    

llm = OpenAI(temperature=0.0)



agent = initialize_agent(tools, 
                         llm, 
                         agent="zero-shot-react-description", 
                         verbose=True)

template = """
    You are a Cybersecurity expert and help me to assist. You have access to the following tools:
    Search: a search engine. Useful for when you need to answer questions. Input should be a query.
    TransfromtoBash: translate the text to bash command to be executed in the VM. Input should be a list of text. Output should be a list of bash commands what will be executed in the VM.
    execute_command: execute commands in the VM. Input should be a list of bash command. the list of commands to be executed one by one in a VM. Output should be a response from the VM's terminal.

"""

#print(agent.agent.llm_chain.prompt.template)

agent.run("Check the app to be updated in the VM , and update it")