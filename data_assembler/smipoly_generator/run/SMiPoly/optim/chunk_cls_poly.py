# import setup for custom smipoly

import json
import os
import sys
from pathlib import Path

import cudf


def extract_type(df):
    # check which column is set with "True" value for a given row for columns other than SMILES and smip_cand_mons
    value_types = [
        "vinyl",
        "epo",
        "cOle",
        "lactone",
        "lactam",
        "hydCOOH",
        "aminCOOH",
        "hindPhenol",
        "cAnhyd",
        "CO",
        "HCHO",
        "sfonediX",
        "BzodiF",
        "diepo",
        "diCOOH",
        "diol",
        "diamin",
        "diNCO",
        "dicAnhyd",
        "pridiamin",
        "diol_b",
    ]
    df["value_type"] = "Unknown"
    for col in value_types:
        df.loc[df[col], "value_type"] = col
    return df


def write_parquet_chunks(df, value_type, chunk_size=1000, metadata={}):
    df_chunk = df[df["value_type"] == value_type]

    # create directory for the specific type
    type_path = f"./datasets/types_{chunk_size}/{value_type}"
    os.makedirs(type_path, exist_ok=True)
    print(f"[+] Directory '{type_path}' created / modified")

    metadata["type_data"][value_type] = {
        "path": type_path,
        "count": len(df_chunk),
        "chunk_paths": [],
    }

    # this chunk can have a lot of rows, so we need to write it in chunks
    for i in range(0, len(df_chunk), chunk_size):
        parquet_path = f"{type_path}/{i}.parquet"
        df_chunk.iloc[i : i + chunk_size].to_parquet(
            parquet_path,
            engine="pyarrow",
            compression="snappy",
        )
        metadata["type_data"][value_type]["chunk_paths"].append(parquet_path)
        print(
            f"[+] Chunk {i} written to file './datasets/types_{chunk_size}/{value_type}/{i}.parquet'"
        )
    return metadata


def main(chunk_size=1000):
    parquet_path = "./datasets/monomer_classified_df.parquet"
    df = cudf.read_parquet(parquet_path)
    print("[+] Read parquet completed")
    # print(df.head())
    # remove the initial unnamed indix column
    df = df.drop(columns=df.columns[0])
    # print(df.columns)

    # apply type extraction for a given row
    df = extract_type(df)
    df["value_type"] = df["value_type"].astype(str)

    # log the value counts output to a log file
    df["value_type"].value_counts().to_frame().to_csv(
        "./datasets/value_type_counts.csv"
    )
    print("[+] Value type counts logged to file './datasets/value_type_counts.csv'")

    # get the dictionary of value_counts to refer
    value_counts_dict = df["value_type"].value_counts().to_dict()

    metadata = {}

    # we need to create a directory to store the types
    os.makedirs(f"./datasets/types_{chunk_size}", exist_ok=True)

    print("[+] Directory './datasets/types' created / modified")

    metadata["base_path"] = f"./datasets/types_{chunk_size}"
    metadata["type_data"] = {}

    for value_type, count in value_counts_dict.items():
        if value_type == "Unknown":
            continue
        meta_data = write_parquet_chunks(df, value_type, chunk_size, metadata)
        print(f"[+] Parquet chunks written for {value_type}")

        # write metadata to a json file
        with open(
            f"./datasets/types_{chunk_size}/{value_type}/metadata.json", "w"
        ) as f:
            json.dump(meta_data, f)

    print("[+] Parquet chunks written")


if __name__ == "__main__":
    CHUNK_SIZE = 10
    main(CHUNK_SIZE)
