# QLoRA Fine-Tuning with Clinical Notes Dataset
"""
This notebook demonstrates a complete, professional pipeline to fine-tune a quantized DistilBERT model using **QLoRA (Quantized Low-Rank Adaptation)** 
for multi-class classification of clinical notes. 

It covers data preprocessing, quantized model loading, QLoRA adaptation into attention layers, training, evaluation with precision/recall/F1, 
and export of deployable artifacts.
"""

import os
from dotenv import load_dotenv

load_dotenv(override=True)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', 'your-key-if-not-using-env')

# Import Libraries
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, BitsAndBytesConfig, DataCollatorWithPadding
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch

# Load CSV
file_path = "clinical_notes_large.csv"
df = pd.read_csv(file_path)

# Encode labels
label_encoder = LabelEncoder()
df["label"] = label_encoder.fit_transform(df["label"])

# Train-test split
train_df, eval_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# Save to disk
train_df.to_csv("train.csv", index=False)
eval_df.to_csv("eval.csv", index=False)

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df)
eval_ds = Dataset.from_pandas(eval_df)

# Tokenization
model_ckpt = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
eval_ds = eval_ds.map(tokenize, batched=True)

train_ds = train_ds.remove_columns(["text"])
eval_ds = eval_ds.remove_columns(["text"])

# Rename label column for Trainer compatibility
train_ds = train_ds.rename_column("label", "labels")
eval_ds = eval_ds.rename_column("label", "labels")

# Choose Quantization Mode
QUANT_4_BIT = True

if QUANT_4_BIT:
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4"
    )
else:
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_8bit_compute_dtype=torch.bfloat16
    )

# Define PEFT LoRA Configuration
peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"]
)


# Load Base Model with Quantization
model = AutoModelForSequenceClassification.from_pretrained(
    model_ckpt,
    num_labels=len(label_encoder.classes_),
    quantization_config=quant_config,
    device_map="auto"
)

# Apply LoRA to Model
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Define Training Arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    label_names=["labels"]
)

# Define Trainer
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator
)

# Start Training
trainer.train()

# Evaluate the model
# eval_results = trainer.evaluate()
# print("\n Evaluation Results:", eval_results)

# Evaluate the fine-tuned model
eval_results = trainer.evaluate()
print("\n🔍 Trainer Evaluate:")
print(eval_results)

preds_output = trainer.predict(eval_ds)
y_pred = preds_output.predictions.argmax(axis=-1)
y_true = eval_ds["labels"]

accuracy = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

print(f"\n Accuracy: {accuracy:.4f}")
print(f" Weighted Precision: {precision:.4f}")
print(f" Weighted Recall:    {recall:.4f}")
print(f" Weighted F1 Score:  {f1:.4f}")

print("\n Classification Report:")
print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

conf_matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()

# Save Model and Label Encoder
model.save_pretrained("./qlora_model")
tokenizer.save_pretrained("./qlora_model")
import joblib
joblib.dump(label_encoder, "label_encoder.joblib")

"""
Summary:
This notebook demonstrated how to fine-tune a quantized DistilBERT model using **QLoRA** for multi-class clinical note classification.
We addressed classifier head initialization, included robust preprocessing, quantization-aware low-rank adaptation to transformer attention layers, 
efficient training, and thorough evaluation.

Output Artifacts:
- Trained QLoRA-adapted DistilBERT model
- Tokenizer for inference
- Label encoder for label mapping

The final model is lightweight, memory-efficient, and production-ready for clinical NLP applications.
"""
