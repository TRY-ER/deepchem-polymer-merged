import json
import os
from heapq import merge
from pathlib import Path

import cudf
import matplotlib.pyplot as plt
from typing_extensions import List


def read_file_map(project_path: str, file_types="all"):
    assert file_types in ["all", "metadata", "parquet"]
    file_map = {}
    # list directories
    for dirs in os.listdir(project_path):
        file_map[dirs] = {}
        if dirs.startswith("."):
            continue
        else:
            # check file type
            base_path = Path(project_path) / dirs
            if file_types == "all":
                file_map[dirs]["meta_files"] = list(Path(base_path).glob("*.json"))
                file_map[dirs]["parquet_files"] = list(
                    Path(base_path).glob("*.parquet")
                )
            elif file_types == "metadata":
                file_map[dirs]["meta_files"] = list(Path(base_path).glob("*.json"))
            elif file_types == "parquet":
                file_map[dirs]["parquet_files"] = list(
                    Path(base_path).glob("*.parquet")
                )

    return file_map


get_valid_name = lambda name: (
    name.split("_")[0] if name.split("_")[-1] == "none" else name
)


def deserialize_meta(file_map):
    type_values = {}
    for dir_name, file_dict in file_map.items():
        name = get_valid_name(dir_name)
        type_values[name] = []
        for file_path in file_dict["meta_files"]:
            with open(file_path, "r") as f:
                data = json.load(f)
                type_values[name].append(data)
    return type_values


def get_type_totals(type_values):
    type_totals = {}
    for type_name, type_data in type_values.items():
        type_totals[type_name] = sum(data["df_shape"][0] for data in type_data)
    return type_totals


def plot_type_totals(type_totals, plot_dir="./plots"):
    plt.bar(type_totals.keys(), type_totals.values())
    plt.xlabel("Types")
    plt.ylabel("Total Rows")
    plt.title("Type Totals")
    # add numbers above the bars
    for i, v in enumerate(type_totals.values()):
        plt.text(i, v + 5, str(v), ha="center", va="bottom")

    # create plot_dir if does not exists
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # save the plot
    plt.savefig(f"{plot_dir}/type_totals.png")


def merge_all_parquet_files(file_map):
    """
    create an empty cudf dataframe. each dir type should be same as
    the type column of the merged dataframe
    """
    merge_df = cudf.DataFrame()
    for dir_name, file_dict in file_map.items():
        reaction_str = ""
        if file_dict["meta_files"]:
            sample_meta = json.loads(file_dict["meta_files"][0])
            assert "rxn_smarts" in sample_meta, (
                f"rxn_smarts not found in {dir_name} json meta"
            )
            reaction_str = sample_meta["rxn_smarts"]
        else:
            print(f"[--] No meta files found in {dir_name}")

        if file_dict["parquet_files"]:
            for file in file_dict["parquet_files"]:
                df = cudf.read_parquet(file)
                df["type"] = get_valid_name(dir_name)
                df["reaction"] = reaction_str
                merge_df = cudf.concat([merge_df, df], ignore_index=True)
        else:
            print(f"[--] No parquet files found in {dir_name}")
    merge_df = merge_df.explode("psmiles")
    return merge_df


def remove_duplicate_psmiles_n_reactions(df):
    """
    remove duplicate rows from the dataframe containing same psmiles and same reaction
    """
    df = df.drop_duplicates(subset=["psmiles", "reaction"])
    return df


if __name__ == "__main__":
    project_path = "../outputs/homo_out/"
    file_map = read_file_map(project_path)

    # this part is for analyzing the each type
    type_values = deserialize_meta(file_map)
    type_totals = get_type_totals(type_values)
    plot_type_totals(type_totals)

    # this part is for dataframe manipulation
    # master_df = merge_all_parquet_files(file_map)
    # master_df = remove_duplicate_psmiles_n_reactions(master_df)
    # print("master_df >> shape:", master_df.shape)
    # print("master_df >> columns:", master_df.columns)
