# data config
BASE_DATA_CONFIG = {
    "data_path": "datasets/finals/dfs/master_df.csv",
    "test_size": 0.2,
    "random_state": 42,
}

SMILES_DATA_CONFIG = {
    "target_column": "smiles",
    **BASE_DATA_CONFIG
}