
import os

# model config for generative transformer
MODEL_CONFIG = {
    "embedding_dim": 128,
    "n_head": 8,
    "n_layers": 6,
    "dropout": 0.1,
    "dim_feedforward": 512,
    # Removed aggregation_type as we now output sequence-level logits
}

# training config
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 1e-4,
    "epochs": 20,
    "patience": 5,
}

# data config
DATA_CONFIG = {
    "data_path": os.path.join("data", "polymers.csv"),
    "smiles_column": "smiles",
    "target_column": "target",
    "test_size": 0.2,
    "random_state": 42,
}
