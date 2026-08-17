# Claims Agent processes structured claims data to extract relevant features and determine high utilization status.
class ClaimsAgent:
    name = "claims_agent"

    def run(self, rows: dict) -> dict:
        c = rows["claim"]
        high_utilization = (
            int(c.get("ed_visits_6m", 0)) >= 3
            or int(c.get("inpatient_admits_6m", 0)) >= 2
        )
        return {
            "ed_visits_6m": int(c.get("ed_visits_6m", 0)),
            "inpatient_admits_6m": int(c.get("inpatient_admits_6m", 0)),
            "specialist_visits_6m": int(c.get("specialist_visits_6m", 0)),
            "allowed_amount_6m": float(c.get("allowed_amount_6m", 0)),
            "high_utilization": high_utilization,
        }
