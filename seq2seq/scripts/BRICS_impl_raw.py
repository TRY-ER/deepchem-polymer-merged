import pandas as pd
import joblib
import BRICSMod as BRICS
from rdkit import Chem
import datetime


def read_df(df_path: str, column_name: str):
    if not df_path.split(".")[-1] == "csv": 
        raise ValueError("The given file path is not targeting a csv fil")
    df = pd.read_csv(df_path)
    if not column_name in df.columns:
        raise ValueError("The given column is not in the df !")
    data = df[column_name].values
    return data

def write_data(data, file_path: str, new_line: bool = True):
    with open(file_path, "w+") as f:
        if type(data) == list: 
            if new_line:
                for d in data:
                    f.write(d+ '\n')
            else:
                f.write(str(data))
        else:
            f.write(data)
    return True

def apply_BRICS(data: list[str]):
    Decompositions = []
    for smiles in data:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"[--] Invalid SMILES >> {smiles}")
        if mol is not None:
            res = BRICS.BRICSDecompose(mol)
            Decompositions += res
    uniques = list(set(Decompositions))
    write_data(uniques, "decomposed_unique.txt")
    print(f"[++] [{datetime.datetime.now()}] Unique blocks found -> {len(uniques)}, write to file -> decomposed_unique.txt") 

    brics_iter = list(BRICS.BRICSBuild(
        [Chem.MolFromSmiles(smiles) for smiles in uniques]
    ))
    text_main = ""
    for i,b in enumerate(brics_iter):
        if type(b) == tuple:
            text, data = b 
            text_main += f"{Chem.MolToSmiles(data)}>>{text} \n"

    print("text main >>", text_main)
    write_data(text_main, "brics_text.txt")
    joblib.dump(brics_iter, "brics_iter.pkl")
    print(f"[++] [{datetime.datetime.now()}] New blocks found -> {len(uniques)}, write to file -> brics_text.txt, brics_iter.pkl") 

if __name__ == "__main__":
    smiles = read_df("./test_df.csv", "SMILES") 
    print(apply_BRICS(smiles))
