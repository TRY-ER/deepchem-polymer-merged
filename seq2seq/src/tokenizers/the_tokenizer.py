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
        if add_special:
            tokens = ["[bos]"] + tokens + ["[eos]"]
        ids = [self.token2idx.get(i, self.token2idx["[unk]"]) for i in tokens]
        if max_length is not None:
            if len(ids) < max_length:
                ids += [self.token2idx["[pad]"]] * (max_length - len(ids))
            else:
                ids = ids[:max_length]
        return ids

    @staticmethod
    def is_valid_token(token):
        return token in base_token_list or token in special_token_list

    @staticmethod
    def is_special_token(token):
        return token in special_token_list

    def decode(self, indices, skip_special=True):
        specials = set(self.special)
        tokens = [self.idx2token.get(i, "[unk]") for i in indices]
        if skip_special:
            tokens = [token for token in tokens if token not in specials]
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
        return self.token2idx["[msk]"]


if __name__ == "__main__":
    tokenizer = TheTokenizer()
    tokens = tokenizer.encode("CCO2_35")
    print(tokens)
    text = tokenizer.decode(tokens, skip_special=False)
    print(tokenizer.encode(text, add_special=False))
    print(text)
