
import os

# model config for generative LSTM
MODEL_CONFIG = {
    "embedding_dim": 128,
    "hidden_dim": 256,  # Increased hidden size for better generation
    "n_layers": 2,
    "dropout": 0.2,
    # Removed bidirectional and aggregation_type as they're not needed for generative models
}

# training config
TRAINING_CONFIG = {
    "batch_size": 32,  # Adjusted for generative training
    "learning_rate": 1e-3,  # Slightly higher learning rate
    "epochs": 10,
    "patience": 5,  # Increased patience
    "grad_clip": 1.0,  # Added gradient clipping
}

# data config
DATA_CONFIG = {
    "data_path": os.path.join("data", "polymers.csv"),
    "smiles_column": "smiles",
    "target_column": "target",
    "test_size": 0.2,
    "random_state": 42,
}
