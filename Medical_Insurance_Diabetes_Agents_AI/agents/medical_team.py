import os
from dotenv import load_dotenv

from agno.agent import Agent
from agno.models.openai import OpenAIChat

from tools.insurance_prediction_tool import predict_insurance_cost


load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "")


def build_medical_insurance_team() -> Agent:
    """
    Compatible version for older Agno versions.

    This uses ONE coordinator agent instead of team agents.
    It avoids:
    - team=
    - show_tool_calls=
    - PDFUrlKnowledgeBase
    - LanceDB
    - Groq
    """

    agent = Agent(
        name="Medical Insurance Diabetes Assistant",
        role=(
            "Help with diabetes-related medical explanation, medical insurance "
            "cost prediction, ML modeling explanation, and safety review."
        ),
        model=OpenAIChat(id="gpt-4o-mini"),
        tools=[predict_insurance_cost],
        description=(
            "You are a healthcare AI assistant for diabetes-related insurance "
            "cost prediction and ML explanation."
        ),
        instructions=[
            "You can answer four types of questions:",
            "1. Diabetes medical education: explain HbA1c, BMI, hypertension, smoking, age, and diabetes risks.",
            "2. Insurance cost prediction: use the predict_insurance_cost tool when all required inputs are provided.",
            "3. ML modeling: explain model design, features, metrics, SHAP, MLflow, deployment, and monitoring.",
            "4. Safety review: ensure no diagnosis, no treatment recommendation, and no final insurance decision.",
            "",
            "Required inputs for prediction are: age, sex, bmi, smoker, children, region, diabetes, hypertension, hba1c.",
            "If prediction inputs are missing, clearly list the missing inputs.",
            "Always explain that predictions are estimates for analytics or educational use only.",
            "Never provide a definitive medical diagnosis.",
            "Never recommend medication changes.",
            "Never make a final insurance approval, denial, or pricing decision.",
            "Use clear sections and concise explanations.",
        ],
        markdown=True,
    )

    return agent