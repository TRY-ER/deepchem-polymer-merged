import os

import cudf

if __name__ == "__main__":
    csv_path = "../monomer_classified_df.csv"
    df = cudf.read_csv(csv_path)
    print("[+] Read CSV Completed")
    parquet_path = "./datasets/monomer_classified_df.parquet"
    # generate path if not exists
    os.makedirs(os.path.dirname(parquet_path), exist_ok=True)
    df.to_parquet(parquet_path)
    print("[+] Write Parquet Completed")
