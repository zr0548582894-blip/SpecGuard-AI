"""SpecGuard AI: baseline requirements-analysis evaluation in LangSmith.

Run from the project directory after configuring .env:
    python evaluate_baseline.py

This evaluates the existing analysis prompt without importing main.py,
which starts an interactive LangGraph run at import time.
"""

import truststore
truststore.inject_into_ssl()

import os
import re
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai.chat_models import GoogleRateLimitError
from google.genai.errors import ServerError
from langsmith import Client
from langsmith.evaluation import evaluate

load_dotenv()

DATASET_NAME = "SpecGuard-AI-Requirements-Baseline-v1"
MODEL_NAMES = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


class RequirementsAnalysis(BaseModel):
    identified_requirements: list[str] = Field(description="Requirements identified in the project")
    missing_information: list[str] = Field(description="Important information missing from the description")
    contradictions: list[str] = Field(description="Contradictions between requirements")
    clarification_question: str = Field(description="One question to clarify the most important missing information")
    needs_clarification: bool = Field(description="Whether clarification is needed before generating the specification")


# A small, synthetic, deliberately unambiguous dataset. Each phrase in
# required_terms/forbidden_terms is a literal token checked in the output.
EXAMPLES = [
    {
        "inputs": {"project_description": "Build a library app. Users can search for books by title and reserve available books.", "analysis_depth": "standard"},
        "outputs": {"required_terms": ["search", "reserv"], "forbidden_terms": ["fine", "payment", "notification"], "has_contradiction": False},
    },
    {
        "inputs": {"project_description": "Create a task manager. Users can create tasks and mark tasks as completed.", "analysis_depth": "standard"},
        "outputs": {"required_terms": ["creat", "complet"], "forbidden_terms": ["reminder", "email", "calendar"], "has_contradiction": False},
    },
    {
        "inputs": {"project_description": "A booking service must allow cancellations at any time. Another requirement states that cancellations are never allowed.", "analysis_depth": "standard"},
        "outputs": {"required_terms": ["cancel"], "forbidden_terms": ["payment", "refund"], "has_contradiction": True},
    },
    {
        "inputs": {"project_description": "An event site must show the event date and location. Users cannot create accounts.", "analysis_depth": "standard"},
        "outputs": {"required_terms": ["date", "location"], "forbidden_terms": ["registration", "login", "password"], "has_contradiction": False},
    },
    {
        "inputs": {"project_description": "A school attendance system must record whether each student is present or absent. The same specification says attendance must never be recorded.", "analysis_depth": "detailed"},
        "outputs": {"required_terms": ["attend", "record"], "forbidden_terms": ["notification", "parent", "grade"], "has_contradiction": True},
    },
]


def analyze_with_fallback(prompt: str) -> RequirementsAnalysis:
    for model_name in MODEL_NAMES:
        try:
            model = ChatGoogleGenerativeAI(model=model_name, temperature=0, max_retries=0)
            return model.with_structured_output(RequirementsAnalysis).invoke(prompt)
        except (ServerError, GoogleRateLimitError) as error:
            if isinstance(error, ServerError) and error.code != 503:
                raise
            print(f"{model_name} unavailable. Trying next model...")
    raise RuntimeError("All Gemini models are currently unavailable.")


def predict(inputs: dict) -> dict:
    # Identical wording to the baseline analyze_project prompt in main.py.
    prompt = f"""
    You are an expert software requirements analyst.
    Project description:
    {inputs['project_description']}
    Required analysis depth:
    {inputs['analysis_depth']}
    Analyze the requirements according to the selected depth.
    Ask a clarification question only if the missing
    information is essential for that depth.
    Do not ask about optional details that can be
    documented as open questions or assumptions.
    If the requirements are sufficiently clear for
    the selected depth, set needs_clarification to False.
    """
    return analyze_with_fallback(prompt).model_dump()


def _contains(text: str, term: str) -> bool:
    return re.search(r"\b" + re.escape(term.lower()) + r"\w*", text.lower()) is not None


def requirement_coverage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    identified = " ".join(outputs.get("identified_requirements", []))
    terms = reference_outputs["required_terms"]
    score = sum(_contains(identified, term) for term in terms) / len(terms)
    return {"key": "requirement_coverage", "score": score}


def unsupported_features(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    # Only evaluate the CONFIRMED requirements list, not open questions.
    identified = " ".join(outputs.get("identified_requirements", []))
    forbidden = reference_outputs["forbidden_terms"]
    hits = [term for term in forbidden if _contains(identified, term)]
    return {"key": "avoids_unsupported_features", "score": 0.0 if hits else 1.0, "comment": f"Forbidden feature terms found: {hits}" if hits else "No listed unsupported features detected"}


def contradiction_detection(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    expected = reference_outputs["has_contradiction"]
    actual = bool(outputs.get("contradictions", []))
    return {"key": "contradiction_detection", "score": 1.0 if expected == actual else 0.0}


def main() -> None:
    if not os.getenv("LANGSMITH_API_KEY"):
        raise RuntimeError("LANGSMITH_API_KEY is missing from .env")
    client = Client()
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"Using existing dataset: {DATASET_NAME}")
    else:
        dataset = client.create_dataset(dataset_name=DATASET_NAME, description="Synthetic cases for baseline SpecGuard requirements-analysis evaluation")
        client.create_examples(
            dataset_id=dataset.id,
            inputs=[example["inputs"] for example in EXAMPLES],
            outputs=[example["outputs"] for example in EXAMPLES],
        )
        print(f"Created dataset: {DATASET_NAME} ({len(EXAMPLES)} examples)")

    print("Running baseline evaluation. This calls Gemini once per dataset example...")
    results = evaluate(
        predict,
        data=DATASET_NAME,
        evaluators=[requirement_coverage, unsupported_features, contradiction_detection],
        experiment_prefix="SpecGuard-Baseline",
        max_concurrency=1,
        client=client,
    )
    print("Evaluation submitted. Open LangSmith > Datasets & Experiments to view results.")
    print("The experiment is ready to inspect in LangSmith.")


if __name__ == "__main__":
    main()
