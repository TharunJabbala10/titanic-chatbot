from __future__ import annotations

import os
from typing import Dict

from app.tools import (
    age_histogram,
    average_fare,
    dataset_columns,
    embarked_counts,
    percent_male,
    survival_by_class,
)

USE_OPENAI = bool(os.getenv("OPENAI_API_KEY"))


def router(question: str) -> Dict:
    q = question.lower()

    # charts
    if ("hist" in q or "histogram" in q) and ("age" in q or "ages" in q):
        return age_histogram()
    if ("survival" in q or "survived" in q) and ("class" in q or "pclass" in q):
        return survival_by_class()

    # text
    if ("percentage" in q or "%" in q) and "male" in q:
        return percent_male()
    if "average" in q and ("fare" in q or "ticket" in q):
        return average_fare()
    if "embark" in q or "port" in q:
        return embarked_counts()
    if "column" in q or "schema" in q:
        return dataset_columns()

    return {
        "type": "text",
        "content": (
            "I can answer Titanic dataset questions and generate charts.\n\n"
            "Try:\n"
            "- What percentage of passengers were male?\n"
            "- What was the average ticket fare?\n"
            "- Show me a histogram of passenger ages\n"
            "- How many passengers embarked from each port?\n"
            "- Show survival rate by class\n"
            "- What columns are in the dataset?"
        ),
    }


# Optional LangChain upgrade (router already satisfies assignment)
if USE_OPENAI:
    try:
        from langchain.agents import AgentType, initialize_agent
        from langchain.tools import Tool
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

        tools = [
            Tool("PercentMale", lambda _: percent_male(), "Percentage of male passengers."),
            Tool("AverageFare", lambda _: average_fare(), "Average ticket fare."),
            Tool("EmbarkedCounts", lambda _: embarked_counts(), "Counts by embarkation port."),
            Tool("AgeHistogram", lambda _: age_histogram(), "Histogram of passenger ages."),
            Tool("SurvivalByClass", lambda _: survival_by_class(), "Survival rate by class."),
            Tool("DatasetColumns", lambda _: dataset_columns(), "List dataset columns."),
        ]

        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )

        def answer(question: str) -> Dict:
            try:
                out = agent.run(question)
                if isinstance(out, dict):
                    return out
                return {"type": "text", "content": str(out)}
            except Exception:
                return router(question)

    except Exception:
        def answer(question: str) -> Dict:
            return router(question)
else:
    def answer(question: str) -> Dict:
        return router(question)