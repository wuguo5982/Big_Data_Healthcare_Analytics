"""Prepare a small NIH ChestX-ray14 image subset for the demo project."""

from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from utils import load_config, project_root, save_json


NIH_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Pleural_Thickening",
    "No Finding",
]


def normalize_label(label: str) -> str:
    """
    Convert a NIH label into a clean column name.

    Example:
    'Pleural Thickening' -> 'pleural_thickening'
    """

    return label.strip().replace(" ", "_").lower()


def find_image_files(raw_image_dir: Path) -> dict[str, Path]:
    """
    Find all NIH image files under the raw image directory.

    Returns:
        A dictionary mapping image filename to full file path.
    """

    image_lookup = {}

    for file_pattern in ["*.png", "*.jpg", "*.jpeg"]:
        for image_path in raw_image_dir.rglob(file_pattern):
            image_lookup[image_path.name] = image_path

    return image_lookup


def load_nih_metadata(csv_path: Path) -> pd.DataFrame:
    """Load NIH ChestX-ray14 metadata CSV."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing NIH metadata CSV: {csv_path}")

    return pd.read_csv(csv_path)


def keep_rows_with_available_images(
    metadata: pd.DataFrame,
    image_lookup: dict[str, Path],
) -> pd.DataFrame:
    """
    Keep only metadata rows that have matching image files.
    """

    matched_metadata = metadata[
        metadata["Image Index"].isin(image_lookup.keys())
    ].copy()

    if matched_metadata.empty:
        raise ValueError(
            "No matching images found between the NIH CSV file and image folder."
        )

    return matched_metadata


def select_demo_sample(
    metadata: pd.DataFrame,
    sample_size: int,
) -> pd.DataFrame:
    """
    Select demo images.

    Images with disease findings are selected first.
    Then images are sorted by image filename for reproducibility.
    """

    metadata = metadata.copy()

    metadata["has_finding"] = (
        metadata["Finding Labels"] != "No Finding"
    ).astype(int)

    selected_metadata = (
        metadata
        .sort_values(
            by=["has_finding", "Image Index"],
            ascending=[False, True],
        )
        .head(sample_size)
    )

    return selected_metadata


def build_image_record(
    row: pd.Series,
    patient_number: int,
    source_image_path: Path,
    new_image_name: str,
) -> dict:
    """
    Build one clean label record for one selected image.
    """

    finding_text = str(row["Finding Labels"])
    finding_set = set(finding_text.split("|"))

    record = {
        "patient_id": f"P{patient_number:05d}",
        "encounter_id": f"E{patient_number:06d}",
        "image_id": new_image_name,
        "original_image_index": row["Image Index"],
        "finding_labels": finding_text,
        "patient_age": row.get("Patient Age", np.nan),
        "patient_gender": row.get("Patient Gender", "Unknown"),
        "view_position": row.get("View Position", "Unknown"),
    }

    for label in NIH_LABELS:
        column_name = normalize_label(label)
        record[column_name] = int(label in finding_set)

    disease_columns = [
        normalize_label(label)
        for label in NIH_LABELS
        if label != "No Finding"
    ]

    record["abnormal_cxr_label"] = int(
        any(record[column] == 1 for column in disease_columns)
    )

    return record


def copy_selected_images_and_create_labels(
    selected_metadata: pd.DataFrame,
    image_lookup: dict[str, Path],
    output_image_dir: Path,
) -> pd.DataFrame:
    """
    Copy selected images to the project folder and create label records.
    """

    output_image_dir.mkdir(parents=True, exist_ok=True)

    label_records = []

    for patient_number, (_, row) in enumerate(
        selected_metadata.iterrows(),
        start=1,
    ):
        original_image_name = row["Image Index"]
        source_image_path = image_lookup[original_image_name]

        new_image_name = (
            f"nih_{patient_number:06d}{source_image_path.suffix.lower()}"
        )

        destination_path = output_image_dir / new_image_name
        shutil.copy(source_image_path, destination_path)

        record = build_image_record(
            row=row,
            patient_number=patient_number,
            source_image_path=source_image_path,
            new_image_name=new_image_name,
        )

        label_records.append(record)

    return pd.DataFrame(label_records)


def save_outputs(
    image_labels: pd.DataFrame,
    labels_output_path: Path,
    annotations_output_path: Path,
) -> None:
    """
    Save image label CSV and project annotation JSON.
    """

    labels_output_path.parent.mkdir(parents=True, exist_ok=True)

    image_labels.to_csv(labels_output_path, index=False)

    save_json(
        {
            "source": "NIH ChestX-ray14",
            "sample_size": len(image_labels),
            "disclaimer": "Research demo only. Not for clinical use.",
        },
        annotations_output_path,
    )

    print(f"Prepared {len(image_labels)} NIH images and labels.")
    print(f"Saved labels to: {labels_output_path}")
    print(f"Saved annotations to: {annotations_output_path}")


def main() -> None:
    """
    Main workflow:
    1. Load project config
    2. Load NIH metadata CSV
    3. Find available raw images
    4. Select a small demo sample
    5. Copy images into the project folder
    6. Save clean image labels
    """

    config = load_config()
    root = project_root()

    nih_csv_path = root / config["nih"]["data_entry_csv"]
    raw_image_dir = root / config["nih"]["raw_dir"]
    output_image_dir = root / config["nih"]["selected_images_dir"]

    labels_output_path = root / config["data"]["image_labels"]
    annotations_output_path = root / config["data"]["annotations"]

    metadata = load_nih_metadata(nih_csv_path)
    image_lookup = find_image_files(raw_image_dir)

    if not image_lookup:
        raise FileNotFoundError(
            f"No NIH image files found under: {raw_image_dir}"
        )

    metadata = keep_rows_with_available_images(
        metadata=metadata,
        image_lookup=image_lookup,
    )

    selected_metadata = select_demo_sample(
        metadata=metadata,
        sample_size=config["project"]["sample_size"],
    )

    image_labels = copy_selected_images_and_create_labels(
        selected_metadata=selected_metadata,
        image_lookup=image_lookup,
        output_image_dir=output_image_dir,
    )

    save_outputs(
        image_labels=image_labels,
        labels_output_path=labels_output_path,
        annotations_output_path=annotations_output_path,
    )


if __name__ == "__main__":
    main()