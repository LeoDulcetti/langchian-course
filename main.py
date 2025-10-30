
from dotenv import load_dotenv
load_dotenv()
import os 
os.environ.pop("LANGSMITH_API_KEY", None)
from langchain_classic.agents import AgentExecutor, create_react_agent# built-in langchain function that is gonna create the ReAct agent for us.
from langchain_classic import hub # hub from langchain is used for sharing prompt and agents created by the community. We are going to use to download a prompt: ReAct Prompt.

# It is a Runnable object, it receives an LLM and a list of tool and a prompt and the prompt we will give it to it is a react prompt.
# An AgentExecutor is a class that is used to execute an agent. It is a Runnable object, it receives an LLM and a list of tool and a prompt and the prompt we will give it to it is a react prompt.
# AgentExecutor is gonna make the actual calls to the LLM and manage the interaction between the LLM and the tools. It is a for loop.
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch 



tools = [TavilySearch()] #This TavilySearch tool is gonna allow the agent to search the web using Tavily API.
# We need to give information to the LLM about how to use this tool. The tool has a name, a description and input.
# There is a schema of the tool like this:
# {
#   "name": "TavilySearch",
#   "description": "Useful for when you need to answer questions about current events or the world. Input should be a search query.",
#   "args": Property that returns the JSON schema of the tool's arguments.
# }

# The model only produces the input for running the tool but it is the AgentExecutor that is gonna actually run the tool with that input and get the output and give it back to the model.

# This TavilySearch has already its description, name and args so we don't need to create a custom tool. 

llm=ChatOpenAI(temperature=0, model_name="gpt-4")
react_prompt= hub.pull("hwchase17/react") # It is a prompt template that is specifically designed for ReAct agents. The template you can check it by debugging the code.
# The ReAct prompt is designed to guide the LLM to reason and act in an interleaved manner. It provides instructions and examples to the model on how to approach tasks using reasoning steps followed by actions.
# It is the basis for the ReAct agent's decision-making process. Check the paper for further information.

agent= create_react_agent(llm, tools, prompt=react_prompt) # It is gonna recive everything we need to create the agent. It is gonna plug all the variables together.
# This is the reasoing engine and now we need the execution of the agent.

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # This is gonna be the agent Runtime. It is gonna manage the interaction between the LLM and the tools.
# It is gonna run a while loop until the agent reaches a final answer. 

chain= agent_executor # The chain is the agent executor. It is a runnable object so we can call it like a function.


def main():
    result= chain.invoke(input={"input": "What is the latest news about LangSmith by LangChain?"}) # The input is the one that is gonna be populated in the prompt template.
    print(result)


if __name__ == "__main__":
    main()


# Tavily helps the agent to connect to the Web API. It has a Free plan. Tavily is gonna expose the services and APIs to the agents.
# It is a langchain provider.