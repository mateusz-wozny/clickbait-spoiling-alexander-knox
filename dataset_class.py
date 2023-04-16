import re
from typing import Dict, List, Optional, Tuple
import pandas as pd
from nltk.stem import WordNetLemmatizer
import torch
import numpy as np


class Dataset:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self.__explode_columns()

    @classmethod
    def from_jsonl(cls, jsonl_path: str):
        return cls(pd.read_json(jsonl_path, lines=True))

    @property
    def df(self):
        return self._df

    def __explode_columns(self):
        col = self._df.columns
        if "tags" in col:
            self._df["tags"] = self._df["tags"].explode()
        if "spoiler" in col:
            self._df["spoiler"] = self._df["spoiler"].apply(lambda x: "\t".join(x))
        if "postText" in col:
            self._df["postText"] = self._df["postText"].explode()

    def count_posts_by_tags(self) -> Dict[str, int]:
        grouped_df = (
            self._df[["uuid", "tags"]]
            .set_index("tags")
            .groupby(level=0)
            .count()
            .reset_index()
        )
        grouped_df.rename(columns={"uuid": "count"}, inplace=True)
        return grouped_df

    def prepare_post_details(
        self, tag: str, left_idx: int = 0, right_idx: Optional[int] = None
    ) -> pd.DataFrame:
        return self._df.loc[
            self._df["tags"] == tag,
            [
                "postText",
                "spoiler",
                "tags",
                "targetTitle",
                "targetParagraphs",
                "spoilerPositions",
            ],
        ].reset_index(drop=True)[left_idx:right_idx]

    def get_spoiler_text_by_position(self, tag_df: pd.DataFrame) -> List[str]:
        spoiler_text = []
        for idx, (paragraph, position) in enumerate(
            zip(tag_df["targetParagraphs"], tag_df["spoilerPositions"])
        ):
            text = ""
            for points in position:
                if points[0][0] == -1:
                    text += tag_df.iloc[idx, 3][points[0][1] : points[1][1]]
                else:
                    text += paragraph[points[0][0]][points[0][1] : points[1][1]]
            spoiler_text.append(text)
        return spoiler_text

    def prepare_data_for_opt_training(self) -> List[dict]:
        data = []
        for row in self.df[
            ["targetParagraphs", "postText", "spoiler", "targetTitle"]
        ].itertuples():
            record = {}
            context = row.targetParagraphs
            clickbait = row.postText
            spoilers = row.spoiler

            record["context"] = " ".join(context)
            record["question"] = clickbait
            record["output"] = spoilers

            data.append(record)
        return data

    def prepare_data_for_fT(self) -> pd.DataFrame:
        fT_df = pd.DataFrame()
        fT_df["uuid"] = self._df["uuid"]
        fT_df["tags"] = self._df["tags"].apply(lambda x: f"__label__{x}")
        fT_df["text"] = (
            self._df["postText"]
            + " - "
            + self._df["targetTitle"]
            + " "
            + self._df["targetParagraphs"].apply(lambda x: " ".join(x))
        )
        return fT_df

    @staticmethod
    def preprocess_func(x: str) -> str:
        stemmer = WordNetLemmatizer()

        document = re.sub(r"\W", " ", x)
        document = re.sub(r"^b\s+", "", document)

        document = document.lower()
        document = document.split()

        document = [stemmer.lemmatize(word) for word in document]
        return " ".join(document)


class TorchDataset(torch.utils.data.Dataset):
    def __init__(
        self, df: pd.DataFrame, labels: dict, tokenizer, max_length: int = 512
    ):
        self.labels = [labels[label] for label in df["tags"]]
        self.texts = [
            tokenizer(
                text,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            for text in df["text"]
        ]

    def classes(self) -> List[str]:
        return self.labels

    def __len__(self) -> int:
        return len(self.labels)

    def get_batch_labels(self, idx: int) -> np.ndarray:
        return np.array(self.labels[idx])

    def get_batch_texts(self, idx: int) -> int:
        return self.texts[idx]

    def __getitem__(self, idx: int) -> Tuple[int, np.ndarray]:
        batch_texts = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)

        return batch_texts, batch_y
