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

portfolio_manager_agent = Agent(
    role="Portfolio Manager",
    goal="Based on a basic idea like {hypothesis} and those of {thought_leaders}, "
        " manage and develop a concrete hypothesis which is falsifiable and objective "
         " that brings together a set of financial factors like {financial_factors} "
         " and trading instruments {instruments}. Also outline"
         "a set of scenarios depending on how different factors play out."
         "Ensure that the final output of the team is ready in the format required in {mental_framework}",
    backstory="You are\ a portfolio manager in a multistrategy hedge fund "
              "specializing in capital and financial markets. This portfolio manager "
              "uses macroeconomic ideas to create a mental model or hypothesis"
                "to identify long term opportunities given current dynamics in the macroeconomic landscape. "
              "When possible, search and identify the key thought leaders and align with the ideas of those "
              "that align well with the hypothesis. You have high mental discipline and insist on the highest standards."
                "You admire Ray Dalio, and Bridgewater. You align with the mental framework required by them.",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool]
)
strategist_analyst_agent = Agent(
    role="Strategist Analyst Agent",
    goal="Flesh out the detailed trading strategies that will be implemented based on how of the  scenarios  provided by the "
        "Portfolio Manager play out: when X happens then do Y.",
    backstory="Equipped with a deep understanding of financial "
              "markets and quantitative analysis, this agent "
              "devises and refines trading strategies based on the PM hypothesis and scenarios."
                " you also flesh out the causal linkages with more finesse and research them on the internet.",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool]
)

trading_strategy_agent = Agent(
    role="Trading Strategy Agent",
    goal="Create very specific the tactical trading playbook to follow in the future"
         "based on the hypothesis, strategies described in {tactical_strategies}, and instruments and mechanisms available.",
    backstory="This agent specializes in providing the playbook describing the timing, tactical steps, "
              "and logistical details of potential trades.",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool]
)

team_editor_agent = Agent(
    role="Team Editor Agent",
    goal="Responsible for ensuring the document follows the requirements outlines in the mental framework  {mental_framework}."
         "Ensure that no content is redundant and that the whole document is self consistent.",
    backstory="Professional editor and proofreader. You are aware of everybody's tasks and ensure that at the end of each review the document aligns with each steps requirements.",
    verbose=True,
    allow_delegation=False
)
risk_management_agent = Agent(
    role="Risk Advisor",
    goal="Evaluate and provide quantitative insights on the risks "
         "associated with potential trading activities: likelihood and VaR.",
    backstory="Armed with a deep understanding of risk assessment models "
              "and market dynamics, this agent will run the scenarios and play out"
              "the risk outcomes (value at risk, drawdown, etc).",
    verbose=True,
    allow_delegation=False,
    tools = [scrape_tool, search_tool]
)

# Task for Strategist Analyst Agent: Analyze Market Data
hypothesis_development_task = Task(
    description=(
        "Conceive and articulate a detailed macroeconomic underlying hypothesis based on the key question of the {hypothesis}"
        "to do this, look at the key ideas of the thought leaders like {thought_leaders}."
        "Consider how the key ideas in the hypothesis drive the dynamics of the financial factors like {financial_factors}.}"
        "Create a set of causal linkages connecting the factors and the instruments specifically {instruments}."
        "Provide a basic hypothesis and outline a set of scenarios that depend on the dynamics that the factors will show in the future."
    ),
    expected_output=(
        "Articulate  a hypothesis as a brief paragraph."
        "Create list of cause and effect linkages between the factors and the instruments."
        "Provide a list of high level scenarios (3 or 4) which are based on the hypothesis, instruments, and causal linkages "
        "presenting alternatives of where the factors will move towards."
    ),
    agent=portfolio_manager_agent,
)

# Task for Trading Strategy Agent: Develop Trading Strategies
trading_strategy_development_task = Task(
    description=(
        "Given a hypothesis, set of instruments, causal linkages and scenarios"
        "articulate a set of high level strategies that align with the horizon in {horizon}"
        "and maintains the level of capital liquidity in {liquidity}. It provides"
        "detailed trading mechanisms that align with the high level mechanisms in {mechanisms}."
        " It also makes use or not of leverage based on the value of the leverage flag {leverage}."
        "Provide high level guidelines of how to act based on how the scenarios play off (e.g., if interest rates increase long DE SWAPS, etc)."
    ),
    expected_output=(
        "A set of potential strategies for the hypotheses developed. "
        "that align with the horizon, risk appetite, mechanisms, instruments, scenarios, and causal linkages."
    ),
    agent=strategist_analyst_agent,
)


# Task for Trade Advisor Agent: Plan Trade Execution
trading_tactics_development_task = Task(
    description=(
        "Create a playbook detailing the checks and mechanisms to action during the horizon in {horizon}"
        "it will detail which instruments to trade, what size of positions to take, when to liquidate, and any other trading attribute, "
        "considering current market conditions and optimal pricing."
    ),
    expected_output=(
        "Create a Detailed execution playbook coming from trading strategy and hypothesis. "
    ),
    agent=trading_strategy_agent,
)


editor_task = Task(
    description=(
        "Editor that ensures quality, consistency, and homogenous document."
    ),
    expected_output=(
        "4 page maximum length document aligning with the mental framework described in {mental_framework}. "
    ),
    agent=team_editor_agent,
)


# Task for Risk Advisor Agent: Assess Trading Risks
risk_guardrails_task = Task(
    description=(
        "Given the hypothesis and scenario provided by the portfolio manager, and the"
        "trading strategy consider the risk of each scenario by calculating the value at"
        "risk, and the likelihood of occurrence and likelihood of outcome for each scenario."
        "Use the risk analysis frameworks in {risk_analysis_frameworks}."
        "Perform drawdawn calculations."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks and mitigation recommendations for hypotheses and scenarios developed"
        "in markdown format."
    ),
    agent=risk_management_agent,
    output_file="L6_final.md"
)

wrap_up_task = Task(
    description=(
        "Given the work of the team, review and improve in a way that aligns with the framework in {mental_framework}."
    ),
    expected_output=(
        "A comprehensive risk analysis report detailing potential "
        "risks and mitigation recommendations for hypotheses and scenarios developed"
        "in markdown format aligning with the framework in {mental_framework}."
    ),
    agent=portfolio_manager_agent,
    output_file="L6_final.md"
)




from crewai import Crew, Process
from langchain_openai import ChatOpenAI

# Define the crew with agents and tasks
financial_trading_crew = Crew(

    agents=[portfolio_manager_agent,
            strategist_analyst_agent,
            trading_strategy_agent,
            risk_management_agent,
            team_editor_agent],

    tasks=[hypothesis_development_task,
           editor_task,
           trading_strategy_development_task,
            editor_task,
           trading_tactics_development_task,
            editor_task,
           risk_guardrails_task,
            editor_task,
           wrap_up_task],

    manager_llm=ChatOpenAI(model="gpt-4o-mini",
                           temperature=0.7),
    process=Process.sequential,
    verbose=True
)

# Example data for kicking off the process
macro_perspective_inputs = {
    'instruments': 'Interest rates; currencies; derivatives' ,
    'hypothesis': 'Modern Mercantilism: The effect of tariffs will introduce a new steady state in global finance. Artificial Intelligence will increase productivity. As described in https://www.bridgewater.com/research-and-insights/modern-mercantilism"]',
    'mechanisms': 'Long; Short; Carry Trades',
    'mental_framework': 'https://www.bridgewater.com/forecasting-the-future-a-modern-economics-challenge',
    'tactical_strategies': "https://www.investopedia.com/terms/t/tactical-trading.asp",
    'thought_leaders': 'Bridgewater Associates in https://www.bridgewater.com/the-growing-risk-to-us-assets-content-ctd',
    'leverage': 'Maximum leverage of 10 percent at any given time.',
    'liquidity': 'No liquidity requirement at any given time.',
    'horizon': 'investment horizon of 24 months',
    'risk_analysis_frameworks': 'https://www.investopedia.com/terms/v/var.asp',
    'financial_factors': 'Tariffs, Global Trade, Capital Investment Flows, Inflation, Interest Rates, Investment in AI',
    'news_impact_consideration': True
}

### this execution will take some time to run
result = financial_trading_crew.kickoff(inputs=macro_perspective_inputs)

print(result)

# Specify the filename
filename = "L6_results.md"
import json
# Open the file in write mode ('w') and save the content
with open(filename, 'w', encoding='utf-8') as f:
    f.write(f"{result.tasks_output}")

print(f"Successfully saved markdown content to {filename}")