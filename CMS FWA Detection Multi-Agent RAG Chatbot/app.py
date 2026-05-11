from flask import Flask, render_template, request, jsonify
from src.config import RAW_DIR
from src.generate_sample_data import generate
from src.data_loader import load_all
from agents.fwa_agents import CMSFWAOrchestrator

app = Flask(__name__)


# Create sample data automatically if the raw CSVs do not exist.
if not (RAW_DIR / "claims.csv").exists():
    generate(str(RAW_DIR))

data = load_all()
orch = CMSFWAOrchestrator(data["claims"], data["providers"], data["notes"])


def _provider_rows_for_ui(n=10):
    """Normalize provider-risk rows for the Level 2 HTML UI."""
    rows = orch.claims_agent.top_providers(n)
    providers = []
    for r in rows:
        providers.append({
            "provider_id": r.get("provider_id"),
            "risk_level": r.get("risk_level"),
            "score": r.get("score", r.get("risk_score")),
            "claims": r.get("claims", r.get("total_claims")),
            "paid_amount": r.get("paid_amount", r.get("total_paid")),
            "reasons": r.get("reasons", "Review recommended")
        })
    return providers


@app.route("/")
def home():
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        payload = request.get_json(force=True) or {}
        user_msg = payload.get("message") or payload.get("query") or ""

        resp = orch.answer(user_msg)

        # Send structured provider rows when the user asks a provider/ranking question.
        route = orch.router.route(user_msg)
        providers = _provider_rows_for_ui(10) if route == "provider_risk" else []

        return jsonify({
            "answer": resp.answer,
            "citations": resp.citations,
            "confidence": resp.confidence,
            "grounding_status": resp.grounding_status,
            "providers": providers
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/provider/<provider_id>", methods=["GET"])
def provider_detail(provider_id):
    try:
        providers = _provider_rows_for_ui(100)
        provider = next((p for p in providers if str(p["provider_id"]).upper() == provider_id.upper()), None)
        if provider is None:
            return jsonify({"error": "Provider not found"}), 404

        # Use the existing analytics agent to retrieve claim-level evidence.
        report = orch.claims_agent.provider_report(provider_id)
        claim_examples = []
        if report:
            _, evidence = report
            for item in evidence:
                claim_examples.append({
                    "claim_id": item.split(":")[0].replace("Claim", "").strip(),
                    "beneficiary_id": "See raw claims data",
                    "procedure_code": item.split("code ")[1].split(",")[0] if "code " in item else "N/A",
                    "paid_amount": item.split("paid $")[1].split(",")[0] if "paid $" in item else "N/A",
                    "flag": item.split("flags=")[-1] if "flags=" in item else "Review"
                })

        provider["claim_examples"] = claim_examples
        return jsonify(provider)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
