"""Generate ClinicalBERT embeddings for each patient."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from utils import load_config, project_root


def mean_pool_embeddings(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Convert token-level BERT outputs into one embedding per clinical document.

    ClinicalBERT returns one vector for each token.
    This function averages only the real tokens and ignores padding tokens.
    """

    mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()

    summed_embeddings = (hidden_states * mask).sum(dim=1)
    token_counts = mask.sum(dim=1).clamp(min=1e-9)

    return summed_embeddings / token_counts


def load_and_combine_clinical_notes(config: dict, root: Path) -> pd.DataFrame:
    """
    Load clinical notes and combine all notes belonging to the same patient.
    """

    documents_path = root / config["data"]["clinical_documents"]
    clinical_notes = pd.read_csv(documents_path)

    combined_notes = (
        clinical_notes
        .groupby("patient_id")["text"]
        .apply(lambda notes: " ".join(notes.astype(str)))
        .reset_index(name="combined_text")
    )

    return combined_notes


def get_device() -> str:
    """
    Use GPU if available; otherwise use CPU.
    """

    if torch.cuda.is_available():
        print("CUDA available: True")
        print("GPU:", torch.cuda.get_device_name(0))
        return "cuda"

    print("CUDA available: False")
    return "cpu"


def generate_clinicalbert_embeddings(
    texts: list[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    """
    Generate one ClinicalBERT embedding for each patient's combined clinical notes.
    """

    all_embeddings = []

    for start_idx in tqdm(range(0, len(texts), batch_size), desc="ClinicalBERT"):
        batch_texts = texts[start_idx : start_idx + batch_size]

        encoded_inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded_inputs = {
            name: value.to(device)
            for name, value in encoded_inputs.items()
        }

        with torch.no_grad():
            model_outputs = model(**encoded_inputs)

        batch_embeddings = mean_pool_embeddings(
            hidden_states=model_outputs.last_hidden_state,
            attention_mask=encoded_inputs["attention_mask"],
        )

        all_embeddings.append(batch_embeddings.cpu().numpy())

    return np.vstack(all_embeddings)


def save_outputs(
    embeddings: np.ndarray,
    patient_ids: pd.DataFrame,
    config: dict,
    root: Path,
) -> None:
    """
    Save ClinicalBERT embeddings and patient index file.
    """

    features_dir = root / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)

    embeddings_path = root / config["data"]["clinicalbert_embeddings"]
    patient_index_path = root / config["data"]["patient_index"]

    np.save(embeddings_path, embeddings)
    patient_ids.to_csv(patient_index_path, index=False)

    print(f"Saved embeddings to: {embeddings_path}")
    print(f"Saved patient index to: {patient_index_path}")
    print("Embedding matrix shape:", embeddings.shape)


def main() -> None:
    """
    Main workflow:
    1. Load config
    2. Load and combine clinical notes by patient
    3. Load ClinicalBERT
    4. Generate embeddings
    5. Save embeddings and patient IDs
    """

    config = load_config()
    root = project_root()
    device = get_device()

    combined_notes = load_and_combine_clinical_notes(config, root)

    tokenizer = AutoTokenizer.from_pretrained(config["clinicalbert"]["model_name"])
    model = AutoModel.from_pretrained(config["clinicalbert"]["model_name"])
    model = model.to(device)
    model.eval()

    embeddings = generate_clinicalbert_embeddings(
        texts=combined_notes["combined_text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        batch_size=config["clinicalbert"]["batch_size"],
        max_length=config["clinicalbert"]["max_length"],
    )

    save_outputs(
        embeddings=embeddings,
        patient_ids=combined_notes[["patient_id"]],
        config=config,
        root=root,
    )


if __name__ == "__main__":
    main()