"""
This file contains the dataloader for the polymer project.
"""

import os
import sys
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach project root
sys.path.insert(0, project_root)

from src.tokenizer import Tokenizer

class PolymerDataset(Dataset):
    """
    A custom dataset for polymer data.
    """
    def __init__(self, smiles_list, penalty_vals: dict[str, list] | None = None):
        self.smiles_list = smiles_list
        self.tokenizer = Tokenizer()
        self.penalty_vals = penalty_vals

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        
        # Tokenize the SMILES string without padding (padding will be done at batch level)
        tokenized_smiles = self.tokenizer.tokenizer(
            smiles, 
            add_special_tokens=True, 
            return_tensors="pt", 
            truncation=True, 
            padding=False  # No padding at individual level
        )
        if self.penalty_vals:
            penalties = {key: torch.tensor(val[idx], dtype=torch.float) for key, val in self.penalty_vals.items()}
            return tokenized_smiles["input_ids"].squeeze(0), penalties 

        return tokenized_smiles["input_ids"].squeeze(0), None  # Remove batch dimension

def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    Pads sequences to the same length within a batch.
    """
    # Extract sequences and penalties from batch
    sequences = []
    penalties = []
    
    for item in batch:
        sequences.append(item[0])  # tokenized sequence
        penalties.append(item[1])  # penalty values (can be None)
    
    # Pad sequences to the same length
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    
    # Handle penalties if they exist
    if penalties[0] is not None:
        # Stack penalty dictionaries into batch format
        batched_penalties = {}
        for key in penalties[0].keys():
            batched_penalties[key] = torch.stack([p[key] for p in penalties])
        return padded_sequences, batched_penalties
    else:
        return padded_sequences, None

def get_dataloaders(DATA_CONFIG, batch_size, PENALTY_CONFIG = None):
    """
    Creates and returns train, validation, and test dataloaders.

    Args:
        DATA_CONFIG (dict): Configuration dictionary containing data paths and parameters.
        batch_size (int): The batch size for the dataloaders.
        PENALTY_CONFIG (dict): Configuration dictionary containing penalty values.

        PENALTY_CONFIG will be of following format
        {
            'penalty_1': 0.5,
            'penalty_2': 0.5,
             ...
        }

    Returns:
        tuple: A tuple containing the train, validation, and test dataloaders.
    """

    assert "data_path" in DATA_CONFIG, "DATA_CONFIG must contain 'data_path'"
    assert "target_column" in DATA_CONFIG, "DATA_CONFIG must contain 'target_column'"
    assert "test_size" in DATA_CONFIG, "DATA_CONFIG must contain 'test_size'"
    assert "random_state" in DATA_CONFIG, "DATA_CONFIG must contain 'random_state'"

    # Load and preprocess data
    df = pd.read_csv(DATA_CONFIG["data_path"])

    # Split data first
    train_df, test_df = train_test_split(
        df, 
        test_size=DATA_CONFIG["test_size"], 
        random_state=DATA_CONFIG["random_state"]
    )

    train_df, val_df = train_test_split(
        train_df, 
        test_size=DATA_CONFIG["test_size"], 
        random_state=DATA_CONFIG["random_state"]
    )

    # Extract penalty data for each split if needed
    train_penalty_data = {}
    val_penalty_data = {}
    test_penalty_data = {}
    
    if PENALTY_CONFIG:
        for key, _ in PENALTY_CONFIG.items():
            assert key in df.columns, f"Penalty key '{key}' not found in dataframe columns"
            train_penalty_data[key] = train_df[key].tolist()
            val_penalty_data[key] = val_df[key].tolist()
            test_penalty_data[key] = test_df[key].tolist()

    # Create datasets
    train_dataset = PolymerDataset(
        smiles_list=train_df[DATA_CONFIG["target_column"]].tolist(),
        penalty_vals=train_penalty_data if len(train_penalty_data) > 0 else None
    )

    val_dataset = PolymerDataset(
        smiles_list=val_df[DATA_CONFIG["target_column"]].tolist(),
        penalty_vals=val_penalty_data if len(val_penalty_data) > 0 else None
    )
    test_dataset = PolymerDataset(
        smiles_list=test_df[DATA_CONFIG["target_column"]].tolist(),
        penalty_vals=test_penalty_data if len(test_penalty_data) > 0 else None
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

# if __name__ == '__main__':
#     # Example usage
#     TEST_CONFIG = {
#         "data_path": "datasets/finals/dfs/p_test.csv",
#         "target_column": "sequence",
#         "test_size": 0.2,
#         "random_state": 42
#     }

#     PENALTY_CONFIG = {
#         "max_C_sequence": 0.5,
#         "sample_penalty": 0.5
#     }

#     train_loader, val_loader, test_loader = get_dataloaders(TEST_CONFIG, batch_size=32, PENALTY_CONFIG=PENALTY_CONFIG)

#     # Print a sample batch
#     for batch in train_loader:
#         print("Sample batch from the dataloader:")
#         sequences, penalties = batch
#         print(f"Sequences shape: {sequences.shape}")
#         print(f"Sequences: {sequences}")
#         if penalties is not None:
#             print(f"Penalties: {penalties}")
#         else:
#             print("No penalties in this batch")
#         break
