import os
import sys

import torch
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.dirname(project_root))

config = {"batch_size": 16, "max_length": 900}
from src.models.diffusion.MDPM import MDPM
from src.tokenizers.the_tokenizer import TheTokenizer


class PolyDiffusionGenerator:
    def __init__(self, tokenizer, meta_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = torch.load(meta_path)
        self.model = MDPM(
            vocab_size=tokenizer.vocab_size,
            pad_id=tokenizer.pad_id,
            mask_id=tokenizer.mask_id,
            T=self.meta["config"]["T"],
            dim=self.meta["config"]["dim"],
            depth=self.meta["config"]["depth"],
            n_heads=self.meta["config"]["n_heads"],
            max_len=self.meta["config"]["max_len"],
            dropout=self.meta["config"]["dropout"],
            cond_dim=self.meta["config"]["cond_dim"],
        ).to(self.device)
        self.tokenizer = tokenizer
        try:
            self.model.load_state_dict(self.meta["model"])
        except FileNotFoundError:
            self.meta = None
            print(f"Meta file not found at {meta_path}")

    @torch.no_grad()
    def generate(
        self,
        n_samples,
        n_steps=1000,
        temparature=1.0,
        top_p=0.9,
    ):
        if self.meta is None:
            raise ValueError("Meta data not loaded")

        # Generate samples using the diffusion model
        token_ids = self.model.sample(
            B=n_samples,
            L=self.meta["config"]["max_len"],
            temparature=temparature,
            n_steps=n_steps,
            top_p=top_p,
            device=self.device,
        )

        results = []
        for i in range(n_samples):
            ids = token_ids[i].tolist()
            result = self.tokenizer.decode(ids, skip_special=True)
            results.append(result)

        return results


if __name__ == "__main__":
    meta_path = "./model_outputs/test_model/best_model.pt"
    tokenizer = TheTokenizer()
    generator = PolyDiffusionGenerator(tokenizer, meta_path)
    samples = generator.generate(n_samples=10, n_steps=100)
    for s in samples:
        print(s, "\n")
