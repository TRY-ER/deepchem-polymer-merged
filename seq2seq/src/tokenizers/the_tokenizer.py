import os
import re
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from src.tokenizers.constants import base_token_list, special_token_list, token_pattern


class TheTokenizer:
    def __init__(self, base=base_token_list, special=special_token_list):
        self.base = base
        self.special = special
        self.token_list = self.special + self.base
        self.token2idx = {token: idx for idx, token in enumerate(self.token_list)}
        self.idx2token = {idx: token for idx, token in enumerate(self.token_list)}

    def encode(self, smiles, add_special=True, max_length=None):
        tokens = re.findall(token_pattern, smiles)
        tokens = [x if x in self.token_list else "[unk]" for x in tokens]
        if add_special:
            tokens = ["[bos]"] + tokens + ["[eos]"]
        if max_length is not None:
            tokens = tokens[:max_length]
        return [self.token2idx[token] for token in tokens]

    @staticmethod
    def is_valid_token(token):
        return token in base_token_list or token in special_token_list

    @staticmethod
    def is_special_token(token):
        return token in special_token_list

    def decode(self, indices, skip_special=True):
        if skip_special:
            indices = [
                idx for idx in indices if not self.is_special_token(self.idx2token[idx])
            ]
        tokens = [self.idx2token[idx] for idx in indices]
        return "".join(tokens)

    @property
    def vocab_size(self):
        return len(self.token_list)

    def __len__(self):
        return self.vocab_size

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.token2idx[key]
        elif isinstance(key, int):
            return self.idx2token[key]
        else:
            raise TypeError("Invalid key type")

    @property
    def pad_id(self):
        return self.token2idx["[pad]"]

    @property
    def bos_id(self):
        return self.token2idx["[bos]"]

    @property
    def eos_id(self):
        return self.token2idx["[eos]"]

    @property
    def unk_id(self):
        return self.token2idx["[unk]"]

    @property
    def mask_id(self):
        return self.token2idx["[mask]"]


if __name__ == "__main__":
    tokenizer = TheTokenizer()
    tokens = tokenizer.encode("CCO2_35")
    print(tokens)
    text = tokenizer.decode(tokens, skip_special=False)
    print(tokenizer.encode(text, add_special=False))
    print(text)
