import sys
import os
import pandas as pd
from rdkit import Chem
sys.path.append(os.path.join(os.path.dirname(__file__),'src'))

from smipoly.smip import polg, monc

def filter_value(value):
    value0 = value.split(">>")[0]
    if value0 == None or value0.strip == "":
        raise ValueError(f"Wrong value found while parsing : {value0}")
    mol = Chem.MolFromSmiles(value0)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {value0}")
    canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
    return canonical_smiles 

if __name__ == "__main__":
    df = pd.read_csv("../sa_filtered_lt_4.csv")
    df["SMILES"] = df["text_data"].apply(filter_value)
    sub_df = pd.read_csv("../202207_smip_monset.csv")
    main_data = [*df["SMILES"].values, *sub_df["SMILES"].values]
    main_df = pd.DataFrame(main_data, columns=["SMILES"])
    main_df = main_df.drop_duplicates(subset=["SMILES"], keep="first")
    print(f"Shape after removing duplicates: {main_df.shape}")
    main_df.to_csv("../smiles_only.csv", index=False)
