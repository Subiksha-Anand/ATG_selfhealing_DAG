import os
os.environ["USE_TF"] = "0"

import transformers
print("Transformers version in use:", transformers.__version__)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset
import torch
from sklearn.metrics import accuracy_score

# Load IMDb dataset
dataset = load_dataset("imdb")

# Use DistilBERT
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Tokenize the data
def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Load model
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

# ✅ Add compute_metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = torch.from_numpy(logits).argmax(dim=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Training configuration
training_args = TrainingArguments(
    output_dir="./checkpoints",
    eval_strategy="epoch",      # ✅ fixed spelling to match argument name
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_dir="./logs"
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(1000)),
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(500)),
    compute_metrics=compute_metrics     # ✅ Important!
)

# Train and save
trainer.train()
trainer.save_model("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
