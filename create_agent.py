from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

from schema import AgentResponse

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o")


# create_agent is a helper function to create an agent with the given model, tools, and response format.
# It is different from the standard LangChain agent creation functions as it allows for a custom response format to be defined using Pydantic models.
# This allows for more structured and validated responses from the agent.

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,
)
# Differently from the create_react_agent function from LangChain, this agent will return structured responses based on the AgentResponse Pydantic model defined in schema.py.
def main():
    result= agent.invoke(
        {"messages": [{"role": "user", "content": "Find me the latest news on AI advancements."}]}
    )
    structured= result.get("structured_response",None)
    print(structured if structured else result)

# We don't need the AgentExecutor from LangChain, as the create_agent function already creates an agent that can be invoked directly. 
# We don't need the Exctract Lambda tool, as the create_agent function already handles the extraction of structured responses based on the Pydantic model.
# LangGraph is gonna be used under the hood to manage the interaction between the model and the tools, but we don't need to interact with it directly.
# It substitues the AgentExecutor from LangChain.
# In order to access the structured response, we can use the "structured_response" key from the result dictionary returned by the agent's invoke method.
if __name__ == "__main__":
    main()