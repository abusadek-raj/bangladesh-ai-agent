import os
from dotenv import load_dotenv
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import TavilySearchResults
from langchain.tools import tool

load_dotenv()

# Initialize Gemini LLM using gemini-2.0-flash
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0
)

# Initialize SQL Databases
db_institutions = SQLDatabase.from_uri("sqlite:///dbs/institutions.db")
db_hospitals = SQLDatabase.from_uri("sqlite:///dbs/hospitals.db")
db_restaurants = SQLDatabase.from_uri("sqlite:///dbs/restaurants.db")

# Helper agents for each database
inst_agent = create_sql_agent(llm=llm, db=db_institutions, agent_type="zero-shot-react-description", verbose=False)
hosp_agent = create_sql_agent(llm=llm, db=db_hospitals, agent_type="zero-shot-react-description", verbose=False)
rest_agent = create_sql_agent(llm=llm, db=db_restaurants, agent_type="zero-shot-react-description", verbose=False)

@tool
def InstitutionsDBTool(query: str) -> str:
    """Useful for querying information about Bangladeshi educational institutions, universities, colleges, and government institutions."""
    res = inst_agent.invoke({"input": query})
    return res["output"]

@tool
def HospitalsDBTool(query: str) -> str:
    """Useful for querying information about hospitals in Bangladesh, including bed capacities, locations, doctors, and facilities."""
    res = hosp_agent.invoke({"input": query})
    return res["output"]

@tool
def RestaurantsDBTool(query: str) -> str:
    """Useful for querying information about restaurants in Bangladesh, including food cuisines, ratings, reviews, and locations."""
    res = rest_agent.invoke({"input": query})
    return res["output"]

# Tavily Web Search Tool
WebSearchTool = TavilySearchResults(
    max_results=3,
    tavily_api_key=os.getenv("TAVILY_API_KEY"),
    description="Useful for answering general knowledge questions, policies, cultural context, or news about Bangladesh."
)