import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cudf
import pandas as pd
from rdkit.Chem import AllChem

root_path = Path(__file__).resolve().parent.parent
src_path = root_path / "src"

if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from smipoly.smip import polg
from smipoly.smip.funclib import bipolymA, genmol, homopolymA


def setup_logger(output_dir="./logs"):
    """Sets up a logger that outputs to both terminal and a file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"exp_{timestamp}.log")

    logger = logging.getLogger("polymerize")
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers
    if logger.handlers:
        return logger

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(log_file)

    # Create formatters and add them to handlers
    format_str = "%(asctime)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(format_str)
    c_handler.setFormatter(formatter)
    f_handler.setFormatter(formatter)

    # Add handlers to the # logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


# Initialize # logger
logger = setup_logger()


def parquet_loader(type_val: str, base_path="./datasets/types", chunk_size=1000):
    # Implement parquet loader logic here
    # check if the directory exists
    type_path = Path(base_path + f"_{chunk_size}" + f"/{type_val}")
    if not type_path.exists():
        raise FileNotFoundError(f"Directory {type_path} does not exist")
    # list  files in the directory
    files = list(type_path.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {type_path}")
    for f in files:
        df = cudf.read_parquet(f)
        yield df


def process_homo(data: tuple):
    assert len(data) == 2, "process_homo expects exactly two arguments"
    smiles, mon_1 = data
    mons = polg.monL[str(polg.mon_dic[mon_1])]
    excls = polg.exclL[str(polg.mon_dic[mon_1])]
    vals = homopolymA(
        mon1=genmol(smiles),
        mons=mons,
        excls=excls,
        targ_mon1=mon_1,
        Ps_rxnL=polg.Ps_rxnL,
        mon_dic=polg.mon_dic,
        monL=polg.monL,
    )
    return vals


def process_bi(data: tuple):
    assert len(data) == 3, "process_bi expects exactly three arguments"
    reactants, targ_rxn, P_class = data
    assert len(reactants) == 2, "process_bi expects exactly two reactants"
    vals = bipolymA(
        reactant=[genmol(reactants[0]), genmol(reactants[1])],
        targ_rxn=targ_rxn,
        monL=polg.monL,
        Ps_rxnL=polg.Ps_rxnL,
        P_class=P_class,
    )
    return vals


def write_exportable_parquet(
    data: list, output_dir: str = "./outputs", inp_type="homo"
):
    j = None
    if inp_type == "homo":
        i, df1_smiles, df2_smiles, df_1_type, df_2_type, psmiles, rxn_smarts = data
    elif inp_type == "hetero":
        i, j, df1_smiles, df2_smiles, df_1_type, df_2_type, psmiles, rxn_smarts = data
    else:
        raise ValueError("Invalid type")

    # input validation setup
    if inp_type == "hetero":
        assert len(df1_smiles) == len(df2_smiles), (
            "df1_smiles and df2_smiles must have the same length"
        )
    assert len(df1_smiles) == len(psmiles), (
        "df1_smiles and psmiles must have the same length"
    )

    # setting up output directory
    type_output_dir = os.path.join(output_dir, f"{df_1_type}_{df_2_type}")
    # check output directory path if exists
    if not os.path.exists(type_output_dir):
        os.makedirs(type_output_dir)

    # write parquet file
    df = pd.DataFrame(
        {
            "df1_smiles": df1_smiles,
            "df2_smiles": ["none"] * len(df1_smiles)
            if df_2_type == "none"
            else df2_smiles,
            "psmiles": psmiles,
        }
    )

    # drop the rows containing empty lists in psmiles column
    df = df[df["psmiles"].apply(lambda x: len(x) > 0)]

    meta_data = {
        "index_1": i,
        "index_2": j if j else "none",
        "rxn_smarts": rxn_smarts,
        "df_1_type": df_1_type,
        "df_2_type": df_2_type,
        "df_shape": df.shape,
        "type": inp_type,
    }

    if inp_type == "homo":
        # write metadata to the output directory for this index
        with open(os.path.join(type_output_dir, f"meta_{i}.json"), "w") as f:
            json.dump(meta_data, f)

        # check if the df_shape contains any data rows
        if df.shape[0] > 0:
            df.to_parquet(os.path.join(type_output_dir, f"polymer_{i}.parquet"))
            logger.info(
                f"Processed homo polymerization for index {i} and wrote to file {os.path.join(type_output_dir, f'polymer_{i}.parquet')}"
            )
        else:
            logger.warning(f"No data rows found for index {i}")
    elif inp_type == "hetero" and j is not None:
        # write metadata to the output directory for this index
        with open(os.path.join(type_output_dir, f"meta_{i}_{j}.json"), "w") as f:
            json.dump(meta_data, f)

        # check if the df_shape contains any data rows
        if df.shape[0] > 0:
            df.to_parquet(os.path.join(type_output_dir, f"polymer_{i}_{j}.parquet"))
            logger.info(
                f"Processed hetero polymerization for index pair of {i} and {j} and wrote to file {os.path.join(type_output_dir, f'polymer_{i}_{j}.parquet')}"
            )
        else:
            logger.warning(f"No data rows found for index pair of {i} and {j}")
    else:
        logger.error(f"Invalid input type: {inp_type}")


def main(specific_targets=None, chunk_size=1000, mode="all"):
    # Determine targets
    targL = []
    if specific_targets is None or specific_targets == ["all"]:
        targL = list(polg.Ps_classL.keys())
    elif specific_targets == ["exc_ole"]:
        targL = list(set(polg.Ps_classL.keys()) - {"polyolefin"})
    else:
        targL = specific_targets

    assert mode in ["all", "homo", "hetero"], f"Invalid mode: {mode}"

    # Collect all monomer classes involved in targets for indexing
    for P_class in targL:
        if str(P_class) in polg.Ps_GenL:
            for P_set in polg.Ps_GenL[str(P_class)]:
                try:
                    if P_set[1] == "none":
                        if mode == "hetero":
                            continue
                        logger.info(f"Processing homo polymerization for {P_set[0]}")
                        # for i, df1 in enumerate(
                        #     parquet_loader(P_set[0], chunk_size=chunk_size)
                        # ):
                        #     df1_smiles = df1["smip_cand_mons"].to_arrow().to_pylist()
                        #     with ThreadPoolExecutor() as executor:
                        #         data = [(smiles, P_set[0]) for smiles in df1_smiles]
                        #         results = list(executor.map(process_homo, data))
                        #     uniques = [list(set(result)) for result in results]
                        #     exportable = [
                        #         i,
                        #         df1_smiles,
                        #         [],
                        #         P_set[0],
                        #         "none",
                        #         uniques,
                        #         AllChem.ReactionToSmarts(P_set[2]),
                        #     ]
                        #     write_exportable_parquet(exportable)
                        #     # break
                    else:
                        if mode == "homo":
                            continue
                        logger.info(
                            f"Starting hetero polymerization for index pair of {P_set[0]} and {P_set[1]}"
                        )
                        for x, df1 in enumerate(
                            parquet_loader(P_set[0], chunk_size=chunk_size)
                        ):
                            df1_smiles = df1["smip_cand_mons"].to_arrow().to_pylist()
                            for y, df2 in enumerate(
                                parquet_loader(P_set[1], chunk_size=chunk_size)
                            ):
                                df2_smiles = (
                                    df2["smip_cand_mons"].to_arrow().to_pylist()
                                )
                                combs = [
                                    [s1, s2]
                                    for s1 in df1_smiles
                                    for s2 in df2_smiles
                                    if s1 != s2
                                ]
                                with ThreadPoolExecutor() as executor:
                                    data = [
                                        (reactants, P_set[2], P_class)
                                        for reactants in combs
                                    ]
                                    results = list(executor.map(process_bi, data))
                                    uniques = [list(set(result)) for result in results]
                                    exportable = [
                                        x,
                                        y,
                                        [x[0] for x in combs],
                                        [x[1] for x in combs],
                                        P_set[0],
                                        P_set[1],
                                        uniques,
                                        AllChem.ReactionToSmarts(P_set[2]),
                                    ]
                                    write_exportable_parquet(
                                        exportable, inp_type="hetero"
                                    )
                                break
                            break
                except FileNotFoundError as f:
                    logger.error(f"File not found: {f}")
                # break
        # break


if __name__ == "__main__":
    SPECIFIC_TARGETS = ["all"]
    CHUNK_SIZE = 10
    MODE = "all"  # this mode can be one of ["all", "homo", "hetero"]
    main(specific_targets=SPECIFIC_TARGETS, chunk_size=CHUNK_SIZE, mode=MODE)

    # create a dummy setup to check the write_exportable_parquet function for hetero case

    # dummy_exportable = [
    #     5,
    #     6,
    #     [
    #         "dummy1",
    #         "dummy2",
    #         "dummy3",
    #         "dummy4",
    #         "dummy5",
    #         "dummy6",
    #         "dummy7",
    #         "dummy8",
    #         "dummy9",
    #         "dummy10",
    #     ],
    #     [
    #         "dummy11",
    #         "dummy12",
    #         "dummy13",
    #         "dummy14",
    #         "dummy15",
    #         "dummy16",
    #         "dummy17",
    #         "dummy18",
    #         "dummy19",
    #         "dummy20",
    #     ],
    #     "test_type_1",
    #     "test_type_2",
    #     [
    #         ["dummy 1_11"],
    #         ["dummy 1_12"],
    #         ["dummy 1_13"],
    #         ["dummy 1_14"],
    #         ["dummy 1_15"],
    #         ["dummy 1_16"],
    #         ["dummy 1_17"],
    #         ["dummy 1_18"],
    #         ["dummy 1_19"],
    #         ["dummy 1_20"],
    #     ],
    #     "some smarts",
    # ]

    # write_exportable_parquet(dummy_exportable, inp_type="hetero")
