from src.generate_sample_data import generate
from src.data_loader import load_all
from agents.fwa_agents import CMSFWAOrchestrator

def test_chatbot_answer(tmp_path):
    generate("data/raw", n_claims=200)
    data = load_all()
    orch = CMSFWAOrchestrator(data["claims"], data["providers"], data["notes"])
    resp = orch.answer("Which providers are most suspicious?")
    assert "potential FWA" in resp.answer or "risk" in resp.answer.lower()
    assert resp.confidence > 0
