from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_classic.agents.output_parsers.react_single_input import ReActSingleInputOutputParser
from typing import Union
from langchain_classic.agents.agent import AgentAction, AgentFinish
from langchain_classic.agents.format_scratchpad import format_log_to_str

from callbacks import AgentCallbackHandler


@tool # A decorator that turns a function into a LangChain tool
def get_text_length(text: str) -> int:
    """Returns the length of the given text by characters"""
    print(f"get_text_length received text: {text}")
    text= text.strip("'\n'").strip('"')  # Remove surrounding quotes if present and stripping non alphanumeric characters
    return len(text)



def find_tool_by_name(tools, tool_name: str):
    for tool in tools:
        if tool.name == tool_name:
            return tool
    raise ValueError(f"Tool with name {tool_name} not found")

if __name__ == "__main__":
    tools= [get_text_length]
    

    template= """ Answer the following questions as best as you can using the provided tools:
    {tools}

    Use the following format:
    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question


    Begin!
    
    Question: {input} 
    Thought: {agent_scratchpad}

    """
    # agent_scratchpad is a special variable that the ReActSingleInputOutputParser will use to insert the previous thoughts, actions, observations, etc. into the prompt for the next step.
    # It is like a memory of what has happened so far in the conversation.
    prompt= PromptTemplate(template=template).partial(
        tools=tools, tool_names=", ".join([t.name for t in tools])) # This "partial" prompt will be completed with input, tools, tool_names
    
    # render_text_description(tools) is a LangChain helper that turns a list of tool objects into a readable text block you can drop straight into a prompt. It enumerates each tool (by name) along with its description (and, in newer versions, a brief view of its args schema), so the model understands what tools exist and when to use them.

    llm= ChatOpenAI(temperature=0, stop=["\nObservation:",  "\n  Observation:", "Observation:"], callbacks=[AgentCallbackHandler()]) # This will tell the LLM to stop generating words and to finish working once it's outputted the backslash observation token.
    # Otherwise the LLM might keep generating text and it's going to guess what comes next, which we don't want.
    # If it comes from an LLM it would be an allusion to an observation, which we don't want.

    intermediate_steps = []

    agent= {"input": lambda x:x['input'], "agent_scratchpad": lambda x: format_log_to_str(x['agent_scratchpad'])} | prompt | llm | ReActSingleInputOutputParser() # Create a simple agent by piping the prompt into the LLM (It is LCEL syntax)
    # The pipe operator takes the output of the left side (the prompt) and feeds it as input to the right side (the LLM)
    # The left side is a function that takes a dictionary and returns the value associated with the 'input' key.
    # When we invoke the chain later with agent.invoke({'input': 'some text'}), the input text will be passed to the prompt.

    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({'input': 'What is the length of the text: "Hello, Leo!"?', 'agent_scratchpad': intermediate_steps})  # Invoke the agent with a sample input question
    print(agent_step) 
    # If agent_step is an instance of AgentAction, it means the agent has decided to take an action.
    # If not, it would be an instance of AgentFinish, indicating the agent has completed its task and provided a final answer.
   

    #  The following code demonstrates how to handle the AgentAction by executing the corresponding tool. Is it because the agent wouldn't able to execute the tool by itself? Answer is yes. Why? Because the agent is just a reasoning engine that decides what to do, but it doesn't have the capability to execute tools directly. The actual execution of tools needs to be handled separately in the code.
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool
        tool_to_use= find_tool_by_name(tools, tool_name)
        tool_input= agent_step.tool_input
        observation= tool_to_use.func(str(tool_input)) # Execute the tool with the provided input
        print(f"Observation: {observation}")
        intermediate_steps.append((agent_step, observation)) # This is needed to keep track of the agent's reasoning process. Agent has both the history of its thoughts and observations. It is like a memory of what has happened so far in the conversation.
        print("Intermediate Steps:", intermediate_steps)
    # The ReActSingleInputOutputParser will parse the LLM's output to extract the final answer after the agent has completed its reasoning and actions.
    # ReActSingleInputOutputParser NO ejecuta nada por sí mismo. Su papel es leer el texto crudo que devuelve el LLM (en el formato ReAct que tú impones en el prompt) y convertirlo en una estructura: { "final_answer": "..." }: 

    # parsed = AgentAction(
    # tool="get_text_length",
    # tool_input='"Hello, World!"',
    # log='Action: get_text_length\nAction Input: "Hello, World!"')


    # class ReActSingleInputOutputParser(name: str | None)
    # Parses ReAct-style LLM calls that have a single tool input.

    # Expects output to be in one of two formats.

    # If the output signals that an action should be taken, should be in the below format. This will result in an AgentAction being returned.

    #Thought: agent thought here
    #Action: search
    #Action Input: what is the temperature in SF?
    # If the output signals that a final answer should be given, should be in the below format. This will result in an AgentFinish being returned.

    #Thought: agent thought here
    #Final Answer: The temperature is 100 degrees
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({'input': 'What is the length of the text: "Hello, Leo!"?', 'agent_scratchpad': intermediate_steps})  # Invoke the agent with a sample input question
    print(f'Second step: {agent_step}') 
    if isinstance(agent_step, AgentFinish):
        print(f"Final Answer: {agent_step.return_values}")
    

    # Finally we invoke the agent again with the updated intermediate steps to get the final answer. The agent will use the observations from the tool execution to arrive at the final answer and the intermediate steps will help it keep track of its reasoning process.

    # Callbacks are a way to hook into the execution of LangChain components to monitor and respond to events that occur during the processing of LLM calls, tool executions, and agent actions.