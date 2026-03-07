import math
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.dirname(project_root))

from src.models.diffusion.datasets import TheDataset
from src.models.diffusion.MDPM import MDPM
from src.tokenizers.the_tokenizer import TheTokenizer

sample_config = {
    "data_path": "../../../datasets/mod/seq2seq_trainer_100_demo.parquet",
    "column_name": "inp_comb_1",
    "max_len": 646,
    "batch_size": 32,
    "T": 1000,
    "dim": 512,
    "depth": 8,
    "n_heads": 8,
    "dropout": 0.1,
    "cond_dim": None,
    "lr": 1e-4,
    "weight_decay": 1e-5,
    "epochs": 10,
    "warmup": 1000,
    "total_steps": 10000,
    "save_dir": "./model_outputs/test_model",
}


class Trainer:
    def __init__(self, config: dict):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.tokenizer = TheTokenizer()
        self.config = config
        self.model = None

    def get_dataloader(self):
        dataset = TheDataset(
            data_path=self.config.get("data_path", "path_to_data"),
            column_name=self.config.get("column_name", "inp_comb_1"),
            max_len=self.config.get("max_len", 646),
        )
        return DataLoader(
            dataset, batch_size=self.config.get("batch_size", 32), shuffle=True
        )

    def lr_lambda(self, step):
        warmup = self.config.get("warmup", 1000)
        if step < warmup:
            return step / warmup
        progress = (step - warmup) / (self.config.get("total_steps", 10000) - warmup)
        return max(
            0.1, 0.5 * (1.0 + torch.cos(torch.tensor(progress * math.pi)).item())
        )

    def train(self):
        # setting up data loader
        data_loader = self.get_dataloader()

        # setting up the model
        self.model = MDPM(
            vocab_size=self.tokenizer.vocab_size,
            pad_id=self.tokenizer.pad_id,
            mask_id=self.tokenizer.mask_id,
            T=self.config.get("T", 1000),
            dim=self.config.get("dim", 512),
            depth=self.config.get("depth", 6),
            n_heads=self.config.get("n_heads", 8),
            max_len=self.config.get("max_len", 646),
            dropout=self.config.get("dropout", 0.1),
            cond_dim=self.config.get("cond_dim", None),
        ).to(self.device)

        print(f"Parameters: {sum(p.numel() for p in self.model.parameters())}")

        # setting up optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.get("lr", 1e-4),
            weight_decay=self.config.get("weight_decay", 1e-5),
        )

        # setting up learing rate
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, self.lr_lambda)
        scaler = torch.amp.GradScaler(str(self.device))

        # training loop
        self.model.train()
        global_step = 0
        best_loss = float("inf")  # Initialize best_loss to infinity

        for epoch in tqdm(range(self.config.get("epochs", 10))):
            epoch_loss = 0.0
            for batch in tqdm(data_loader):
                x0 = batch["x0"].to(self.device)
                cond = (
                    batch["cond"].to(self.device)
                    if self.config.get("cond_dim", None) is not None
                    else None
                )

                if cond is not None and torch.rand(1).item() < 0.1:
                    cond = None

                optimizer.zero_grad()
                with torch.amp.autocast(str(self.device)):
                    loss = self.model(x0, cond=cond)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler_before = scaler.get_scale()
                scaler.step(optimizer)

                scaler.update()
                if scaler.get_scale() == scaler_before:
                    scheduler.step()

                epoch_loss += loss.item()
                global_step += 1

            avg_epoch_loss = epoch_loss / len(data_loader)
            print(f"Epoch {epoch} Avg Loss: {avg_epoch_loss}")

            # Save only the best model checkpoint
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                save_dir = self.config.get("save_dir", "./model_output")
                os.makedirs(save_dir, exist_ok=True)
                print(
                    f"New best model found with loss: {best_loss:.4f}. Saving checkpoint..."
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "config": self.config,
                        "global_step": global_step,
                        "model": self.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "best_loss": best_loss,
                    },
                    f"{save_dir}/best_model.pt",
                )
        return self.model, self.tokenizer


if __name__ == "__main__":
    run_config = {
        "data_path": "../../../datasets/mod/seq2seq_trainer_100_demo.parquet",
        "column_name": "inp_comb_1",
        "max_len": 646,
        "batch_size": 8,
        "T": 1000,
        "dim": 512,
        "depth": 4,
        "n_heads": 8,
        "dropout": 0.1,
        "cond_dim": None,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "epochs": 10,
        "warmup": 1000,
        "total_steps": 10000,
        "save_dir": "./model_outputs/test_model",
    }
    trainer = Trainer(run_config)
    model, tokenizer = trainer.train()
