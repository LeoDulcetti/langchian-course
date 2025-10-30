
from dotenv import load_dotenv
load_dotenv()
import os 
os.environ.pop("LANGSMITH_API_KEY", None)
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_classic import hub
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch 
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
# RunnableLambda is a class that allows you to create a runnable object from a lambda function.RunnableLambda allows to integrate logic defined in a lambda function into a runnable pipeline and invoke it like any other runnable object in LangChain.


from prompt import REACT_PROMPT_WITH_FORMAT_INSTRUCTIONS
from schemas import AgentResponse
tools = [TavilySearch()] 
llm=ChatOpenAI(temperature=0, model_name="gpt-4")

structured_llm= llm.with_structured_output(AgentResponse) # It is going to create a new LLM that is gonna produce structured output according to the AgentResponse schema defined in schemas.py.
# Same as doing before PydanticOutputParser but this way is more straightforward when you just want to have the LLM producing structured output according to a pydantic schema.
react_prompt= hub.pull("hwchase17/react") 
agent= create_react_agent(llm, tools, prompt=react_prompt) 
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) 
# Here
extract_output= RunnableLambda(lambda x: x['output']) # It is going to extract the output from the agent executor result. If the agent executor returns a dictionary with multiple keys, we just want the output key.
chain= agent_executor | extract_output | structured_llm # The chain is the agent executor piped to the extract output and then piped to the structured LLM.
# The result of the agent executor is passed to the extract output which extracts the output and then the extracted output is passed to the structured LLM which parses it into the AgentResponse schema.


def main():
    result= chain.invoke(input={"input": "What is the latest news about LangSmith by LangChain?"}) 

if __name__ == "__main__":
    main()
