"""Train a multimodal fusion model using EHR + ClinicalBERT + chest X-ray images."""

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from utils import load_config, project_root


EHR_COLUMNS = [
    "age",
    "bmi",
    "systolic_bp",
    "diastolic_bp",
    "heart_rate",
    "spo2",
    "resp_rate",
    "temperature_f",
    "creatinine",
    "glucose",
    "bnp",
    "troponin",
    "wbc",
    "lvef",
    "qrs_ms",
    "qt_ms",
    "comorbidity_count",
    "abnormal_cxr_label",
]


class MultimodalPatientDataset(Dataset):
    """
    PyTorch dataset for one patient encounter.

    Each sample contains:
    - structured EHR features
    - ClinicalBERT text embeddings
    - chest X-ray image
    - high-risk label
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        image_dir,
        bert_columns: list[str],
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.bert_columns = bert_columns

        self.image_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.dataframe)

    def __getitem__(self, index: int):
        row = self.dataframe.iloc[index]

        image_path = self.image_dir / row["image_id"]
        image = Image.open(image_path).convert("RGB")
        image = self.image_transform(image)

        ehr_features = torch.tensor(
            row[EHR_COLUMNS].values.astype(np.float32)
        )

        bert_features = torch.tensor(
            row[self.bert_columns].values.astype(np.float32)
        )

        label = torch.tensor(
            float(row["high_risk_label"]),
            dtype=torch.float32,
        )

        return ehr_features, bert_features, image, label


class MultimodalFusionModel(nn.Module):
    """
    Fusion model that combines:
    - image features from ResNet18
    - structured EHR features
    - ClinicalBERT text embeddings
    """

    def __init__(self, ehr_input_dim: int, bert_input_dim: int):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.image_encoder = nn.Sequential(
            *list(resnet.children())[:-1]
        )

        for parameter in self.image_encoder.parameters():
            parameter.requires_grad = False

        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.20),
        )

        self.text_encoder = nn.Sequential(
            nn.Linear(bert_input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.20),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 + 64 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        ehr_features: torch.Tensor,
        bert_features: torch.Tensor,
        image: torch.Tensor,
    ) -> torch.Tensor:
        image_features = self.image_encoder(image)
        image_features = image_features.view(image_features.size(0), -1)

        ehr_features = self.ehr_encoder(ehr_features)
        text_features = self.text_encoder(bert_features)

        combined_features = torch.cat(
            [image_features, ehr_features, text_features],
            dim=1,
        )

        logits = self.classifier(combined_features)

        return logits.squeeze(1)


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


def load_multimodal_data(config: dict, root) -> tuple[pd.DataFrame, list[str]]:
    """
    Load EHR data, patient index, and ClinicalBERT embeddings.
    Then merge them into one dataframe.
    """

    ehr_data = pd.read_csv(root / config["data"]["structured_ehr"])
    patient_index = pd.read_csv(root / config["data"]["patient_index"])
    bert_embeddings = np.load(root / config["data"]["clinicalbert_embeddings"])

    bert_columns = [
        f"bert_{i}"
        for i in range(bert_embeddings.shape[1])
    ]

    bert_df = pd.DataFrame(bert_embeddings, columns=bert_columns)
    bert_df["patient_id"] = patient_index["patient_id"]

    multimodal_df = ehr_data.merge(bert_df, on="patient_id")

    return multimodal_df, bert_columns


def scale_ehr_features(df: pd.DataFrame, root) -> pd.DataFrame:
    """
    Standardize numeric EHR features and save the scaler.
    """

    scaler = StandardScaler()
    df[EHR_COLUMNS] = scaler.fit_transform(df[EHR_COLUMNS])

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    scaler_path = model_dir / "multimodal_ehr_scaler.pkl"
    joblib.dump(scaler, scaler_path)

    print("Saved EHR scaler to:", scaler_path)

    return df


def create_data_loaders(
    df: pd.DataFrame,
    image_dir,
    bert_columns: list[str],
    batch_size: int,
):
    """
    Split data into train/test sets and create PyTorch DataLoaders.
    """

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        stratify=df["high_risk_label"],
        random_state=42,
    )

    train_dataset = MultimodalPatientDataset(
        dataframe=train_df,
        image_dir=image_dir,
        bert_columns=bert_columns,
    )

    test_dataset = MultimodalPatientDataset(
        dataframe=test_df,
        image_dir=image_dir,
        bert_columns=bert_columns,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader, train_df


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
) -> dict:
    """
    Evaluate the model on the test set.
    """

    model.eval()

    probabilities = []
    labels = []

    with torch.no_grad():
        for ehr_features, bert_features, image, label in data_loader:
            ehr_features = ehr_features.to(device)
            bert_features = bert_features.to(device)
            image = image.to(device)

            logits = model(ehr_features, bert_features, image)
            batch_probabilities = torch.sigmoid(logits).cpu().numpy()

            probabilities.extend(batch_probabilities)
            labels.extend(label.numpy())

    probabilities = np.array(probabilities)
    labels = np.array(labels)

    predictions = (probabilities >= 0.50).astype(int)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "f1": f1_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, probabilities),
        "pr_auc": average_precision_score(labels, probabilities),
    }


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    loss_function,
    optimizer,
    device: str,
    epoch_number: int,
) -> None:
    """
    Train the model for one epoch.
    """

    model.train()

    for ehr_features, bert_features, image, label in tqdm(
        train_loader,
        desc=f"Epoch {epoch_number}",
    ):
        ehr_features = ehr_features.to(device)
        bert_features = bert_features.to(device)
        image = image.to(device)
        label = label.to(device)

        optimizer.zero_grad()

        logits = model(ehr_features, bert_features, image)
        loss = loss_function(logits, label)

        loss.backward()
        optimizer.step()


def save_metrics(metrics: dict, root) -> None:
    """
    Save final multimodal fusion metrics to CSV.
    """

    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / "multimodal_fusion_metrics.csv"

    if not metrics:
        raise ValueError("No metrics were generated. Evaluation may not have run.")

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    print("Saved multimodal fusion metrics to:", metrics_path)


def main() -> None:
    """
    Main workflow:
    1. Load EHR data
    2. Load ClinicalBERT embeddings
    3. Merge EHR + text data
    4. Load chest X-ray images
    5. Train multimodal fusion model
    6. Save best model and final metrics
    """

    config = load_config()
    root = project_root()
    device = get_device()

    data, bert_columns = load_multimodal_data(config, root)

    data = scale_ehr_features(data, root)

    image_dir = root / config["nih"]["selected_images_dir"]

    train_loader, test_loader, train_df = create_data_loaders(
        df=data,
        image_dir=image_dir,
        bert_columns=bert_columns,
        batch_size=config["model"]["batch_size"],
    )

    model = MultimodalFusionModel(
        ehr_input_dim=len(EHR_COLUMNS),
        bert_input_dim=len(bert_columns),
    ).to(device)

    positive_count = train_df["high_risk_label"].sum()
    negative_count = len(train_df) - positive_count

    positive_weight = negative_count / max(positive_count, 1)

    loss_function = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(
            [positive_weight],
            dtype=torch.float32,
            device=device,
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["model"]["learning_rate"],
    )

    model_dir = root / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_roc_auc = -1
    final_metrics = {}

    for epoch in range(1, config["model"]["epochs"] + 1):
        train_one_epoch(
            model=model,
            train_loader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            device=device,
            epoch_number=epoch,
        )

        metrics = evaluate_model(model, test_loader, device)
        print(metrics)

        if metrics["roc_auc"] > best_roc_auc:
            best_roc_auc = metrics["roc_auc"]

            best_model_path = model_dir / "best_multimodal_fusion_model.pt"
            torch.save(model.state_dict(), best_model_path)

            print("Saved best model to:", best_model_path)

        final_metrics = metrics

    save_metrics(final_metrics, root)


if __name__ == "__main__":
    main()