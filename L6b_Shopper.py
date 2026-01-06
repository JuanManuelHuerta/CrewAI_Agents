# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew

import os

import os
from dotenv import load_dotenv
load_dotenv()

openai_api_key = os.getenv('OPENAI_API_KEY')
os.environ["OPENAI_MODEL_NAME"] = 'gpt-4o-mini'
os.environ["SERPER_API_KEY"] =os.getenv('SERPER_API_KEY')


from crewai_tools import ScrapeWebsiteTool, SerperDevTool

search_tool = SerperDevTool()
scrape_tool = ScrapeWebsiteTool()

shopper_agent = Agent(
    role="Shopper",
    goal="Based on a target ASIN id ({target_asin}) and a reference ASIN ({reference_asin}), "
        "perform product comparisons. ",
    backstory="You are meticulous, careful professional shopper. ",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool]
)

# Task for Strategist Analyst Agent: Analyze Market Data
product_reasoning_task = Task(
    description=(
        "Identify the key product specifications for the target and reference asins {target_asin}  and {reference_asin}, respectively. "
        " Identify the common set of product specifications for the reference and for the target and produce output based on these common product specifications. "
    ),
    expected_output=(
        "Return a json object with the key value pairs for the target product attribute."
        "Also include a price_justification attribute with possible values (high, medium, low) and "
        "a reasoning field wht a string stating succintly if the price is justified or not."
    ),
    agent=shopper_agent,
)


from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# Define the crew with agents and tasks
shopper_crew = Crew(

    agents=[shopper_agent],

    tasks=[product_reasoning_task],

    manager_llm=ChatOpenAI(model="gpt-4o-mini",
                           temperature=0.7),

    process=Process.sequential,
    verbose=True
)

# Example data for kicking off the process
shopping_inputs = {
    'target_asin': 'B00NGVF4II',
    'reference_asin': 'B0C544R2J7'
}

### this execution will take some time to run
result = shopper_crew.kickoff(inputs=shopping_inputs)

print(result)

# Specify the filename
filename = "L6b_results.md"
import json
# Open the file in write mode ('w') and save the content
with open(filename, 'w', encoding='utf-8') as f:
    f.write(f"{result.tasks_output}")

print(f"Successfully saved markdown content to {filename}")