from src.data_preparation import load_claims
from src.config import CLAIMS_PATH


def test_load_claims():
    df = load_claims(CLAIMS_PATH)
    assert len(df) == 1000
    assert "claim_amount" in df.columns
    assert "label" in df.columns
