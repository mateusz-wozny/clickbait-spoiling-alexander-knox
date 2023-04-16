from datasets import Dataset
from transformers import AutoTokenizer
from evaluate import load
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np


def preprocess_function(examples, tokenizer):
    return tokenizer(examples["text"], truncation=True)


def prepare_model_and_data(
    model_checkpoint: str, train_dataset: Dataset, val_dataset: Dataset
):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True)
    train_dataset = train_dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
    val_dataset = val_dataset.map(
        lambda x: preprocess_function(x, tokenizer), batched=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_checkpoint, num_labels=3
    )
    return model, tokenizer, train_dataset, val_dataset


metric = load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def train(
    model_checkpoint,
    train_dataset,
    val_dataset,
    batch_size=8,
    lr=2e-5,
    epochs=7,
    **kwargs,
):
    model, tokenizer, train_dataset, val_dataset = prepare_model_and_data(
        model_checkpoint, train_dataset, val_dataset
    )

    metric_name = "accuracy"
    model_name = model_checkpoint.split("/")[-1]
    args = TrainingArguments(
        f"{model_name}-finetuned",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=10,
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        push_to_hub=False,
        **kwargs,
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    return trainer
