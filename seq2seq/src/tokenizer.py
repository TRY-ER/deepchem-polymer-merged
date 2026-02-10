import torch
import pandas as pd
from transformers import BertTokenizer 

class Tokenizer:
    def __init__(self, model_name='bert-base-cased'):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(self, smiles_list):
        inputs = self.tokenizer(smiles_list, add_special_tokens = True, return_tensors="pt", truncation=True, padding=True)
        return inputs["input_ids"]

if __name__ == '__main__':
    # Example Usage
    tokenizer = Tokenizer()
    print("tokenizer vocal size >>",tokenizer.tokenizer.vocab_size)
    
    # Load your data
    df = pd.read_csv('../datasets/finals/dfs/filtered_sequences_case_study_sample_1K.csv')
    sample_smiles = df['sequence'].head(10).tolist()
    
    print(f"Original SMILES: {sample_smiles}")
    
    # Get embeddings
    embeddings = tokenizer.tokenize(sample_smiles)
    
    print(f"Shape of the returned tensor: {embeddings.shape}")
    print("type of the returned tensor:", type(embeddings))
    print(f"Embeddings for the first 3 SMILES strings: {embeddings}")
