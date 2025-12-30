import os
from crewai import Agent, Task, Crew
from instructor.cli.batch import results

from dotenv import load_dotenv
load_dotenv()


os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_MODEL_NAME"] =  os.getenv("OPENAI_MODEL_NAME")

info_agent=Agent(
    role="Information Agent",
    goal="give compelling information about a certain topic",
    backstory="""
    You love to know information. People love and hate you for it. You win most of the quizzes at 
    your local pub.
    """

)


task_1=Task(
    description="Tell me all about the blue ring octopus.",
    expected_output="Give me a quick summary and then 7 bullet points describing it.",
    agent=info_agent

)

crew=Crew(
    agents=[info_agent],
    tasks=[task_1],
    verbose=True
)

result = crew.kickoff()
print("######")
print(result)

