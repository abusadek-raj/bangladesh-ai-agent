import os
from dotenv import load_dotenv
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from tools import InstitutionsDBTool, HospitalsDBTool, RestaurantsDBTool, WebSearchTool

load_dotenv()

# Setup Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Tool registry
tools = [
    InstitutionsDBTool,
    HospitalsDBTool,
    RestaurantsDBTool,
    WebSearchTool
]

# Standard ReAct Prompt Template
react_prompt_template = """Answer the following questions as best you can. You have access to the following tools:

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
Thought:{agent_scratchpad}"""

prompt = PromptTemplate.from_template(react_prompt_template)

# Build ReAct Agent
agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

# Build Agent Executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
)

def run_query(query: str):
    print(f"\n--- User Query: {query} ---")
    response = agent_executor.invoke({"input": query})
    print(f"\nFinal Response:\n{response['output']}")
    return response['output']

if __name__ == "__main__":
    # Test Data/Statistics query
    run_query("How many hospitals are in Dhaka?")
    
    # Test Web Search query
    # run_query("What is the role of DGHS in Bangladesh?")