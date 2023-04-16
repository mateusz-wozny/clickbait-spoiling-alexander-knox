from typing import List, Optional, Tuple
import torch
from transformers import BloomTokenizerFast, BloomForCausalLM
import pandas as pd
from dataset_class import Dataset
from tqdm import tqdm


def load_model() -> Tuple[BloomTokenizerFast, BloomForCausalLM, torch.device]:
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    tokenizer = BloomTokenizerFast.from_pretrained("bigscience/bloom-560m")
    model = BloomForCausalLM.from_pretrained("bigscience/bloom-560m")
    model.to(device)
    return tokenizer, model, device


def get_prompt(examples_per_class: int) -> str:
    datapath = "data/train.jsonl"
    dataset = Dataset.from_jsonl(datapath)
    df = pd.DataFrame()
    df["text"] = (
        "Question: \n"
        + dataset.df["postText"]
        + "\nContext: \n"
        + dataset.df["targetParagraphs"].apply(lambda x: ". ".join(x)[:2500])
        + "\nAnswer: \n"
        + dataset.df["spoiler"]
    )
    df["tags"] = dataset.df["tags"]
    train_df = pd.concat(
        [
            df[df["tags"] == "phrase"].sample(examples_per_class),
            df[df["tags"] == "passage"].sample(examples_per_class),
            df[df["tags"] == "multi"].sample(examples_per_class),
        ]
    )
    train_df = train_df.sample(frac=1)
    prompt = "\n\n".join(train_df["text"])
    return prompt


def generate_spoilers(
    df: pd.DataFrame,
    examples_per_class: int,
    number_of_generated_spoilers: Optional[int] = None,
) -> List[str]:
    spoilers = []
    tokenizer, model, device = load_model()
    prompt = get_prompt(examples_per_class=examples_per_class)
    if number_of_generated_spoilers:
        number_of_generated_spoilers = df.shape[0]

    for i in tqdm(range(number_of_generated_spoilers)):
        sample = df.iloc[i, :]
        sample_text = sample["text"]
        test_prompt = prompt + "\n\n" + sample_text
        input_ids = tokenizer(test_prompt, return_tensors="pt").to(device)
        output = model.generate(**input_ids, max_new_tokens=40, top_k=1)
        spoiler = tokenizer.decode(output[0]).split(sample_text)
        words = [spoiler[1] if len(spoiler) > 1 else " "][0].split("\n")[0].split()
        spoilers.append(" ".join(sorted(set(words), key=words.index)))
    return spoilers
