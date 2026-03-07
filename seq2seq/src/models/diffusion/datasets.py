import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.dirname(project_root))

from src.tokenizers.the_tokenizer import TheTokenizer


class TheDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        column_name: str = "smiles",
        tokenizer=None,
        max_len: int = 700,
        properties=None,
    ):
        self.tokenizer = tokenizer or TheTokenizer()
        self.data_path = data_path
        self.data = self.load_data(column_name)
        self.props = properties
        self.max_len = max_len

    def load_data(self, column_name):
        if self.data_path.split(".")[-1] == "csv":
            df = pd.read_csv(self.data_path)
        elif self.data_path.split(".")[-1] == "json":
            df = pd.read_json(self.data_path)
        elif self.data_path.split(".")[-1] == "parquet":
            df = pd.read_parquet(self.data_path)
        else:
            raise ValueError(
                f"Unsupported file format: {self.data_path.split('.')[-1]}"
            )
        if column_name in df.columns:
            return df[column_name].values
        raise ValueError(f"The column name {column_name} does not exist in dataset !")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        tokens = self.tokenizer.encode(seq, add_special=True, max_length=self.max_len)
        item = {"x0": torch.tensor(tokens, dtype=torch.long)}
        if self.props is not None:
            props = self.props[idx]
            item["cond"] = torch.tensor(props, dtype=torch.float32)
        return item


if __name__ == "__main__":
    # for column inp_comb_1 the max_len is 645
    # for column inp_comb_0 the max_len is 995
    dataset = TheDataset(
        data_path="../../../datasets/mod/seq2seq_trainer_100_demo.parquet",
        column_name="inp_comb_1",
        max_len=646,
    )
    print(len(dataset))
    print(dataset[0])
