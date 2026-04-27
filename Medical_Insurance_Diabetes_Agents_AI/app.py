
from agents.medical_team import build_medical_insurance_team


def main() -> None:
    agent_team = build_medical_insurance_team()
    query = """
    Patient profile:
    age=61
    sex=male
    bmi=30.15
    smoker=yes
    children=1
    region=northeast
    diabetes=yes
    hypertension=yes
    hba1c=8.20

    Estimate annual medical insurance cost and explain:
    1. Why the predicted cost may be high
    2. Which factors matter most
    3. Which ML model is suitable
    4. What safety limitations should be considered
    """
    agent_team.print_response(query, stream=True)


if __name__ == "__main__":
    main()
