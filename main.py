
import truststore
truststore.inject_into_ssl()
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Command
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from google.genai.errors import ServerError
from langchain_google_genai.chat_models import GoogleRateLimitError

class ProjectState(TypedDict):
    project_description: str
    clarification_count: int
    needs_clarification: bool
    status: str
    result: str
    identified_requirements: list[str]
    missing_information: list[str]
    contradictions: list[str]
    clarification_question: str
    analysis_depth: str
    revision_request: str
    
class RequirementsAnalysis(BaseModel):
    identified_requirements: list[str] = Field(
        description="Requirements identified in the project"
    )
    missing_information: list[str] = Field(
        description="Important information missing from the description"
    )
    contradictions: list[str] = Field(
        description="Contradictions between requirements"
    )
    clarification_question: str = Field(
        description="One question to clarify the most important missing information"
    )
    needs_clarification: bool = Field(
        description="Whether clarification is needed before generating the specification"
    )

load_dotenv()

MODEL_NAMES = [
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite"
]


def analyze_with_fallback(prompt):
    for model_name in MODEL_NAMES:
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                max_retries=0
            )

            analyzer = model.with_structured_output(
                RequirementsAnalysis
            )

            return analyzer.invoke(prompt)

        except (ServerError, GoogleRateLimitError) as error:
            if isinstance(error, ServerError) and error.code != 503:
                raise

            print(f"{model_name} unavailable. Trying next model...")

    raise RuntimeError("All Gemini models are currently unavailable.")

def analyze_project(state: ProjectState):
    print("\nAnalyzing requirements...")

    prompt = f"""
    You are an expert software requirements analyst.

    Project description:
    {state["project_description"]}

    Required analysis depth:
    {state["analysis_depth"]}

    Analyze the requirements according to the selected depth.

    Only include requirements explicitly stated by the user
    or directly supported by the project description.

    Do not invent, assume, or add common software features
    that the user has not requested.

    If a potentially important feature is not confirmed,
    do not include it in identified_requirements.
    If essential, document it in missing_information
    and consider asking a clarification question.

    Distinguish confirmed requirements from missing
    information and assumptions.

    Identify contradictions between stated requirements
    without resolving them by inventing new requirements.

    Ask a clarification question only if the missing
    information is essential for that depth.

    Do not ask about optional details that can be
    documented as open questions or assumptions.

    If the requirements are sufficiently clear for
    the selected depth, set needs_clarification to False.
    """

    analysis = analyze_with_fallback(prompt)

    return analysis.model_dump()

def choose_next_step(state: ProjectState):
    if  state["needs_clarification"]:
        return "ask_clarification"

    return "generate_spec"

def ask_clarification(state: ProjectState):
    answer = interrupt(
        state["clarification_question"]
    )
    if answer.strip().lower() == "done":
        return {
            "needs_clarification": False,
            "status": "clarification_finished"
        }
    return {
        "project_description": (
            state["project_description"] + "\n" + answer
        ),
        "clarification_count": state["clarification_count"] + 1,
        "status": "clarified"
    }

def after_clarification(state: ProjectState):
    if state["status"] == "clarification_finished":
        return "generate_spec"

    return "analyze_project"

def generate_with_fallback(prompt):
    for model_name in MODEL_NAMES:
        try:
            model = ChatGoogleGenerativeAI(
                model=model_name,
                temperature=0,
                max_retries=0
            )
            response = model.invoke(prompt)
            return response.text

        except (ServerError, GoogleRateLimitError) as error:
            if isinstance(error, ServerError) and error.code != 503:
                raise

            print(f"{model_name} unavailable. Trying next model...")

    raise RuntimeError("All Gemini models are currently unavailable.")

def generate_spec(state: ProjectState):
    print("\nGenerating specification...")

    if state["revision_request"]:
        prompt = f"""
        You are a professional software requirements analyst.

        Update the existing specification.

        Existing specification:
        {state["result"]}

        User's requested changes:
        {state["revision_request"]}

        Preserve all existing requirements and test cases
        unless the user explicitly asks to change them.

        Apply only the requested changes.
        Do not invent new requirements.

        Return the complete updated specification.
        """

    else:
        prompt = f"""
        You are a professional software requirements analyst.

        Create a software requirements specification
        based on this project description:

        {state["project_description"]}

        Include these sections:
        1. Project overview
        2. User roles
        3. Functional requirements
        4. Business rules
        5. Missing information and open questions
        6. Test cases

        Previously identified missing information:
        {state["missing_information"]}

        Previously identified contradictions:
        {state["contradictions"]}

        Use only requirements explicitly provided by the user.

        Do not turn common features or assumptions
        into confirmed functional requirements.

        If a feature was not explicitly requested,
        list it as an open question instead.

        Clearly distinguish confirmed requirements
        from missing or unconfirmed information.

        Write in clear, professional English.
        """

    specification = generate_with_fallback(prompt)

    return {
        "status": "spec_generated",
        "result": specification,
        "revision_request": ""
    }

def review_spec(state: ProjectState):
    feedback = interrupt(
        "Type 'approve' to accept the specification, "
        "or describe the changes you want."
    )

    if feedback.strip().lower() == "approve":
        return {"status": "approved"}

    return {
        "status": "revision_requested",
        "revision_request": feedback
    }

def after_review(state: ProjectState):
    if state["status"] == "approved":
        return "end"

    return "generate_spec"

builder = StateGraph(ProjectState)

builder.add_node("analyze_project", analyze_project)
builder.add_node("ask_clarification", ask_clarification)
builder.add_node("generate_spec", generate_spec)
builder.add_node("review_spec", review_spec)

builder.add_edge(START, "analyze_project")
builder.add_conditional_edges(
    "analyze_project",
    choose_next_step,
    {
        "ask_clarification": "ask_clarification",
        "generate_spec": "generate_spec"
    }
)
builder.add_conditional_edges(
    "ask_clarification",
    after_clarification,
    {
        "analyze_project": "analyze_project",
        "generate_spec": "generate_spec"
    }
)
builder.add_edge("generate_spec", "review_spec")
builder.add_conditional_edges(
    "review_spec",
    after_review,
    {
        "generate_spec": "generate_spec",
        "end": END
    }
)

graph = builder.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "project_1"}}

initial_state = {
    "project_description": "Library management system",
    "clarification_count": 0,
    "needs_clarification": False,
    "status": "new",
    "result": "",
    "identified_requirements": [],
    "missing_information": [],
    "contradictions": [],
    "clarification_question": "",
    "analysis_depth": "standard",
    "revision_request": "",
}

result = graph.invoke(initial_state, config=config)

while "__interrupt__" in result:
    message = result["__interrupt__"][0].value

    if result.get("status") == "spec_generated":
        print("\nSpecification for review:\n")
        print(result["result"])

    print("\n", message)

    user_answer = input("Your answer: ")

    result = graph.invoke(
        Command(resume=user_answer),
        config=config
    )

print("\nFinal status:", result["status"])