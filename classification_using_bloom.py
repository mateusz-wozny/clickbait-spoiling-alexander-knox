from typing import Dict, List, Optional, Tuple
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, classification_report

from dataset_class import Dataset


def load_model() -> Tuple[BloomTokenizerFast, BloomForCausalLM, torch.device]:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
    model.to(device)
    return tokenizer, model, device


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_datapath = "data/train.jsonl"
    train_dataset = Dataset.from_jsonl(train_datapath)
    train_df = pd.DataFrame()
    train_df["text"] = "\nQuestion:\n" + train_dataset.df["postText"]
    train_df["tags"] = train_dataset.df["tags"]

    test_datapath = "data/validation.jsonl"
    test_dataset = Dataset.from_jsonl(test_datapath)
    test_df = pd.DataFrame()
    test_df["text"] = "\nQuestion:\n" + test_dataset.df["postText"]
    test_df["tags"] = test_dataset.df["tags"]
    return train_df, test_df


def classify_spoiler_type(
    examples_per_class: int, number_of_test_examples: Optional[int] = None
) -> List[str]:
    tokenizer, model, device = load_model()

    train_df, test_df = load_data()
    train_df = pd.concat(
        [
            train_df[train_df["tags"] == "phrase"].head(examples_per_class),
            train_df[train_df["tags"] == "passage"].head(examples_per_class),
            train_df[train_df["tags"] == "multi"].head(examples_per_class),
        ]
    )
    train_df = train_df.sample(frac=1)
    prompt = train_df["text"] + "\nType of answer: " + train_df["tags"]

    predicted_class = []
    if number_of_test_examples is None:
        number_of_test_examples = test_df.shape[0]

    for i in range(number_of_test_examples):
        test_prompt = "\n ".join(prompt.values)
        sample = test_df.iloc[i, :]
        test_prompt += "\n" + sample["text"] + "\nType of answer: "
        input_ids = tokenizer(test_prompt, return_tensors="pt").to(device)
        output = model.generate(**input_ids, max_new_tokens=1, top_k=1)
        predicted_class.append(
            tokenizer.decode(output[0]).rsplit("Type of answer: ", 1)[1].strip()
        )
    return predicted_class


def calculate_metrics(test_df: pd.DataFrame, predicted_class: List[str]) -> Dict:
    labels = {"phrase": 0, "passage": 1, "multi": 2}
    test_df["tags"] = test_df["tags"].apply(lambda x: labels[x])
    test_df["predicted_tag"] = list(map(lambda x: labels.get(x, 0), predicted_class))

    acc = balanced_accuracy_score(test_df["tags"], test_df["predicted_tag"])
    report = classification_report(test_df["tags"], test_df["predicted_tag"])

    return {"accuracy": acc, "metrics_per_class": report}
