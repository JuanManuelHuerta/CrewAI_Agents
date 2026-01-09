# Warning control
import warnings
warnings.filterwarnings('ignore')

from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from crewai.tools import BaseTool
from typing import List, Dict

from crewai_tools import DirectoryReadTool, \
                         FileReadTool


#target_asin="B00NGVF4II"
target_asin="B0B1GZMSCV"

directory_read_tool = DirectoryReadTool(directory=f"./product_data/{target_asin}")
file_read_tool = FileReadTool()
file_read_tool_masked = FileReadTool(file_path=f"product_data/{target_asin}/masked.json")

class SentimentAnalysisTool(BaseTool):
    name: str = "Sentiment Analysis Tool"
    description: str = ("Analyzes the sentiment of text "
                        "to ensure positive and engaging communication.")

    def _run(self, text: str) -> str:
        # Your custom code tool goes here
        return "positive"



class ProductAttributes(BaseModel):
    """Output from first task"""
    products: List[str]
    attributes: List[str]

class ProductData(BaseModel):
    """Individual product with its data"""
    product_id: str
    price: float
    attributes: Dict[str, str]  # attribute_name: attribute_value

class ProductDataList(BaseModel):
    """Output from second task"""
    products: List[ProductData]
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
    goal="Based on a target ASIN id  and a set of cohort ASINs, "
        "retrieve information and find a set of common product attributes. ",
    backstory="You are meticulous, careful professional analyst shopper. ",
    verbose=True,
    allow_delegation=False,
    output_pydantic=ProductAttributes,
    tools = [scrape_tool, search_tool]
)

data_collector = Agent(
    role='Data Collector',
    goal='Collect detailed product information including prices and attribute values',
    backstory='Meticulous researcher who gathers comprehensive product data',
    verbose=True
)


data_masker = Agent(
    role='Data Masker',
    goal='Perform masking operations on product data.',
    backstory='Fidelity in passthrough data with randomized masking as described in task.',
    verbose=True
)

price_estimator = Agent(
    role='Price Estimator',
    goal='Determines if a specific price is competitive and compares versus other products.',
    backstory='Rational and Evidence based estimation.',
    verbose=True
)



# Task for Strategist Analyst Agent: Analyze Market Data
task1 = Task(
    description=(
        "Retrieve the detailed product specifications for query asin {target_asin} "
        " then search for a list of comparable products, and  using their ASIN strings save these and the target ASIN as a list of ASIN ids (products). "
        " Finally using  the key product specifications for the target and cohort asins  find the set of 5 to 10 attributes whose values are likely to determine product price for this cohort and target asin. "
        " Please include brand, and material, if available."
    ),
    expected_output="Find the list of 8 to 10 ASINs of comparable products (products) including the target ASIN and identify 5 to 10 product features that is common across the cohort (attributes). Do not add any comments.",
    agent=shopper_agent,
    output_json=ProductAttributes,
    tools = [scrape_tool, search_tool],
    output_file=f"product_data/{target_asin}/cohort_characteristics.json"
)

task2 = Task(
    description="""
    For each product identified in the previous task, create a detailed record with:
    - product_id: A unique identifier for the product
    - price: The current market price (in USD)
    - attributes: A dictionary mapping each attribute name to its value for this product

    Use the products list and attributes list from the previous task.
    Ensure every product has values for all attributes identified.
    """,
    agent=data_collector,
    expected_output='Complete product data with prices and attribute values for all products',
    output_pydantic=ProductDataList,
    context=[task1]  # This gives task2 access to task1's output
)

task3 = Task(
description="""
     Receive the ProductDataList json object from the previous task and pick a product at random then mask the price of that 
    product (i.e., replace the price with an -1.0 value (to keep it consistent with the type)). 
         """,
    agent=data_masker,
    expected_output='Masked product data with prices and attribute values for all products. Do not add any comments.',
    output_pydantic=ProductDataList,
    output_file=f"product_data/{target_asin}/masked.json",
    context=[task2]  # This gives task2 access to task1's output



)

task4 = Task(
description="""
     Receives the set of products and from it reasons if the price of the query product {target_asin} is competitive.
     To determine this, it uses the other products, their prices, and their features to explain if the price is competitive or
     too high or too low and why.
        """,
    agent=price_estimator,
    context=[task2],
    output_file=f"product_data/{target_asin}/price_reasoning.md",
    expected_output="For the product {target_asin} provide first a determination if the price is high, fair, or low. Then explain why using other products prices and features. Use markdown."

)

from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# Define the crew with agents and tasks
shopper_crew = Crew(

    agents=[shopper_agent, data_collector, data_masker, price_estimator],

    tasks=[task1, task2, task3, task4],

    manager_llm=ChatOpenAI(model="gpt-4o-mini",
                           temperature=0.7),

    process=Process.sequential,
    verbose=True
)

# Example data for kicking off the process
shopping_inputs = {
    'target_asin': target_asin
}

### this execution will take some time to run
result = shopper_crew.kickoff(inputs=shopping_inputs)



print("Task 1 Output:")
print(f"Products: {task1.output.pydantic.products}")
print(f"Attributes: {task1.output.pydantic.attributes}")

print("\nTask 2 Output:")
for product in task2.output.pydantic.products:
    print(f"\nProduct: {product.product_id}")
    print(f"Price: ${product.price}")
    print(f"Attributes: {product.attributes}")

'''

print(result)

# Specify the filename
filename = "L6b_results.md"
import json
# Open the file in write mode ('w') and save the content
with open(filename, 'w', encoding='utf-8') as f:
    f.write(f"{result.tasks_output}")

print(f"Successfully saved markdown content to {filename}")

'''