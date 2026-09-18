import os
import json

from google import genai

from tools import (
    profile_dataset,
    descriptive_statistics,
    correlation_analysis,
    train_model,
)

# -----------------------------------------
# GEMINI CLIENT
# -----------------------------------------

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL = "gemini-3.6-flash"


# -----------------------------------------
# AGENT PLANNER
# -----------------------------------------


def create_plan(profile, target, user_goal):

    prompt = f"""
You are an autonomous Data Science Agent.

Your job is to examine a dataset profile and
decide which Data Science tools should be used
to answer the user's request.

DATASET PROFILE:

{json.dumps(profile, indent=2)}

TARGET COLUMN:

{target}

USER GOAL:

{user_goal}


AVAILABLE TOOLS:

1. profile_dataset
   Understand the structure of the dataset.

2. descriptive_statistics
   Calculate statistical summaries.

3. correlation_analysis
   Find relationships between numerical variables.

4. train_model
   Train a machine-learning model using the
   selected target variable.


RULES:

- Choose only useful tools.
- Do NOT choose train_model if there is no target.
- Do not invent columns.
- Return ONLY valid JSON.

Example:

{{
    "plan": [
        "descriptive_statistics",
        "correlation_analysis",
        "train_model"
    ]
}}
"""

    interaction = client.interactions.create(model=MODEL, input=prompt)

    text = interaction.output_text.strip()

    # Remove markdown code fences if Gemini adds them

    text = text.replace("```json", "")

    text = text.replace("```", "")

    return json.loads(text)


# -----------------------------------------
# EXECUTE AGENT PLAN
# -----------------------------------------


def run_agent(df, target, user_goal):

    # First understand the dataset

    profile = profile_dataset(df)

    # Ask Gemini to create a plan

    plan = create_plan(profile, target, user_goal)

    results = {}

    # Execute the tools selected by the agent

    for tool in plan["plan"]:

        if tool == "profile_dataset":

            results["profile"] = profile

        elif tool == "descriptive_statistics":

            results["statistics"] = descriptive_statistics(df)

        elif tool == "correlation_analysis":

            results["correlations"] = correlation_analysis(df)

        elif tool == "train_model":

            if target:

                results["model"] = train_model(df, target)

    return plan, results


# -----------------------------------------
# GENERATE FINAL REPORT
# -----------------------------------------


def generate_report(results):

    prompt = f"""
You are a senior Data Scientist.

Interpret the following results generated
by Python Data Science tools.

RESULTS:

{json.dumps(
    results,
    indent=2,
    default=str
)}


Create a concise professional report.

Include:

1. Key findings
2. Important relationships
3. Machine-learning performance
4. Possible explanations
5. Three recommended next analyses

IMPORTANT:

- Do not invent information.
- Do not claim correlation proves causation.
- Clearly distinguish observations from
  possible explanations.
"""

    interaction = client.interactions.create(model=MODEL, input=prompt)

    return interaction.output_text
