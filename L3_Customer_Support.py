import warnings
warnings.filterwarnings('ignore')

import os
from crewai import Agent, Task, Crew
from instructor.cli.batch import results

from dotenv import load_dotenv
load_dotenv()



from crewai_tools import SerperDevTool, ScrapeWebsiteTool,  WebsiteSearchTool



support_agent = Agent(
    role="Senior Consultant",
	goal="Be the most informative, valuable and helpful "
        "AI consultant in your team",
	backstory=(
		"You work at a large Consulting Firm (BCG, PwC) and "
        " are now working on providing "
		" consultative insights {customer}, a super important customer "
        " for your company."
		"You need to make sure that you provide the best  consulting insights."
		"Make sure to provide full complete analyses, "
        " and make no assumptions."
	),
	allow_delegation=False,
	verbose=True
)

support_quality_assurance_agent = Agent(
	role="AI Specialist",
	goal="Get recognition for providing the "
    "best Consulting Work in your team",
	backstory=(
		"You are a Senior Manager at a top consulting firm and "
        "are now working with your team "
		"on a request from {customer} ensuring that "
        "the support representative is "
		"providing the best support possible.\n"
		"You need to make sure that the senior consultant "
        "is providing full"
		"complete answers, and make no assumptions."
	),
	verbose=True
)

docs_scrape_tool = ScrapeWebsiteTool(
    website_url="https://www.langchain.com/stateofaiagents"
)

inquiry_resolution = Task(
    description=(
        "{customer} just reached out with a super important ask:\n"
	    "{inquiry}\n\n"
        "{person} from {customer} is the one that reached out. "
		"Make sure to use everything you know "
        "to provide the best answer possible."
		"You must strive to provide a complete "
        "and accurate response to the customer's inquiry."
    ),
    expected_output=(
	    "A detailed, informative response to the "
        "customer's inquiry that addresses "
        "all aspects of their question.\n"
        "The response should include references "
        "to everything you used to find the answer, "
        "including external data or solutions. "
        "Ensure the answer is complete, "
		"leaving no questions unanswered, and maintain a helpful and friendly "
		"tone throughout."
    ),
	tools=[docs_scrape_tool],
    agent=support_agent,
)


quality_assurance_review = Task(
    description=(
        "Review the response drafted by the Consultant for {customer}'s inquiry. "
        "Ensure that the answer is comprehensive, accurate, and adheres to the "
		"high-quality standards expected for customer support.\n"
        "Verify that all parts of the customer's inquiry "
        "have been addressed "
		"thoroughly, with a helpful and friendly tone.\n"
        "Check for references and sources used to "
        " find the information, "
		"ensuring the response is well-supported and "
        "leaves no questions unanswered."
    ),
    expected_output=(
        "A final, detailed, and informative response "
        "ready to be sent to the customer.\n"
        "This response should fully address the "
        "customer's inquiry, incorporating all "
		"relevant feedback and improvements.\n"
		"Don't be too formal, we are a chill and cool company "
	    "but maintain a professional and friendly tone throughout."
    ),
    agent=support_quality_assurance_agent,
)

crew = Crew(
  agents=[support_agent, support_quality_assurance_agent],
  tasks=[inquiry_resolution, quality_assurance_review],
  memory=True
)


inputs = {
    "customer": "Zinnia",
    "person": "Juan Huerta",
    "inquiry": "Give me a summary of the current LangChain state of AI Agents"
}
result = crew.kickoff(inputs=inputs)
print(result)




