import argparse
import concurrent.futures
import itertools
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

# Add package root to path to allow relative imports
script_path = Path(__file__).resolve()
package_root = script_path.parent.parent.parent
if str(package_root) not in sys.path:
    sys.path.append(str(package_root))

from smipoly.smip import polg
from smipoly.smip.funclib import bipolymA, genmol, homopolymA


def load_chunk(path):
    """Loads a chunk file (parquet or csv)."""
    path = Path(path)
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    else:
        return pd.read_csv(path)


def build_chunk_index(metadata_file, target_classes):
    """
    Scans all chunks to determine which monomer classes are present in which chunks.
    Returns a dictionary mapping class_name -> list of chunk paths.
    """
    logging.info("Indexing chunks to optimize iteration...")
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    base_dir = Path(metadata_file).parent
    chunk_index = {cls: [] for cls in target_classes}

    # Also need to track columns available in the dataset
    available_classes = set()

    for file_info in metadata["files"]:
        chunk_path = base_dir / file_info["filename"]
        try:
            # We only need to check if columns have any True values
            # Reading entire file might be slow if chunks are huge, but necessary to check content.
            # Optimization: Read only class columns if possible, but pandas reads all by default unless specified.
            # parquet allows reading specific columns.

            if chunk_path.suffix == ".parquet":
                # Get columns first
                import pyarrow.parquet as pq

                pq_file = pq.ParquetFile(chunk_path)
                cols = pq_file.schema.names

                # Identify which target classes are in columns
                relevant_cols = [c for c in target_classes if c in cols]
                available_classes.update(relevant_cols)

                if not relevant_cols:
                    continue

                # Read only relevant columns
                df = pd.read_parquet(chunk_path, columns=relevant_cols)
            else:
                df = pd.read_csv(chunk_path)
                available_classes.update([c for c in target_classes if c in df.columns])

            for cls in target_classes:
                if cls in df.columns:
                    # Check if any row is True (handling string 'False'/'True' if csv)
                    if df[cls].dtype == object:
                        has_true = (
                            df[cls]
                            .astype(str)
                            .replace({"False": False, "True": True})
                            .any()
                        )
                    else:
                        has_true = df[cls].any()

                    if has_true:
                        chunk_index[cls].append(str(chunk_path))

        except Exception as e:
            logging.warning(f"Failed to index chunk {chunk_path}: {e}")
    logging.info("Indexing complete.")
    for cls, paths in chunk_index.items():
        logging.info(f"Class '{cls}': found in {len(paths)} chunks.")

    return chunk_index


def process_chunks(input_dir, output_dir, specific_targets=None, max_workers=None):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Configure logging
    log_file = output_path / "process.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
        force=True,
    )

    if max_workers is None:
        max_workers = os.cpu_count() or 4
    logging.info(f"Using {max_workers} workers for processing.")

    metadata_path = input_path / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: metadata.json not found in {input_dir}")
        return

    # Setup Polg variables
    # Replicate filtering from polg.biplym
    monL_filtered = {
        k: v
        for k, v in polg.monLg.items()
        if k in polg.mon_vals[0] + polg.mon_vals[1] + polg.mon_vals[2]
    }
    exclL_filtered = {
        k: v
        for k, v in polg.exclLg.items()
        if k in polg.mon_vals[0] + polg.mon_vals[1] + polg.mon_vals[2]
    }

    # Determine targets
    targL = []
    if specific_targets is None or specific_targets == ["all"]:
        targL = list(polg.Ps_classL.keys())
    elif specific_targets == ["exc_ole"]:
        targL = list(set(polg.Ps_classL.keys()) - {"polyolefin"})
    else:
        targL = specific_targets

    # Collect all monomer classes involved in targets for indexing
    needed_classes = set()
    for P_class in targL:
        if str(P_class) in polg.Ps_GenL:
            for P_set in polg.Ps_GenL[str(P_class)]:
                needed_classes.add(P_set[0])  # targ_mon1
                if P_set[1] != "none":
                    needed_classes.add(P_set[1])  # targ_mon2

    chunk_index = build_chunk_index(metadata_path, list(needed_classes))

    # Iterate over targets
    total_polymers = 0
    tasks_submitted = 0
    tasks_completed = 0

    # Bound the number of pending futures to avoid memory issues
    max_pending_tasks = max_workers * 4
    pending_futures = set()

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for P_class in targL:
            logging.info(f"Processing Polymer Class: {P_class}")

            if str(P_class) not in polg.Ps_GenL:
                continue

            for set_idx, P_set in enumerate(polg.Ps_GenL[str(P_class)]):
                targ_mon1 = P_set[0]
                targ_mon2 = P_set[1]
                targ_rxn = P_set[2]

                # Find Rxn Key
                Ps_rxnL_key = [
                    k
                    for k, v in polg.Ps_rxnL.items()
                    if AllChem.ReactionToSmarts(v) == AllChem.ReactionToSmarts(targ_rxn)
                ][0]

                logging.info(f"  Reaction Set {set_idx}: {targ_mon1} + {targ_mon2}")

                chunks1 = chunk_index.get(targ_mon1, [])
                if not chunks1:
                    logging.info(f"    No candidates for {targ_mon1}. Skipping.")
                    continue

                # Generator for task arguments
                def task_arg_generator():
                    if targ_mon2 == "none":
                        # Homopolymerization
                        for i, chunk_path in enumerate(chunks1):
                            yield (chunk_path, None, f"set{set_idx}_c{i}")
                    else:
                        # Copolymerization
                        chunks2 = chunk_index.get(targ_mon2, [])
                        if not chunks2:
                            logging.info(
                                f"    No candidates for {targ_mon2}. Skipping."
                            )
                            return

                        total_pairs = len(chunks1) * len(chunks2)
                        logging.info(f"    Processing {total_pairs} chunk pairs...")
                        '''
                        The above part basically requires a new layer of chunking and processing. instead loading it entirely
                        The chunking has to be done from a dataset layer for optimization.
                        '''

                        for i, c1_path in enumerate(chunks1):
                            for j, c2_path in enumerate(chunks2):
                                yield (c1_path, c2_path, f"set{set_idx}_c{i}_c{j}")

                for c1_path, c2_path, batch_id in task_arg_generator():
                    while len(pending_futures) >= max_pending_tasks:
                        done, pending_futures = concurrent.futures.wait(
                            pending_futures,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        for future in done:
                            try:
                                count = future.result()
                                if count:
                                    total_polymers += count
                            except Exception as e:
                                logging.error(f"Task failed: {e}")
                            tasks_completed += 1
                            if tasks_completed % 100 == 0:
                                logging.info(
                                    f"Completed {tasks_completed} tasks. Total polymers so far: {total_polymers}"
                                )

                    future = executor.submit(
                        process_batch,
                        c1_path,
                        c2_path,
                        targ_mon1,
                        targ_mon2,
                        P_class,
                        Ps_rxnL_key,
                        targ_rxn,
                        monL_filtered,
                        exclL_filtered,
                        polg.mon_dic,
                        polg.Ps_rxnL,
                        output_dir,
                        batch_id,
                    )
                    pending_futures.add(future)
                    tasks_submitted += 1

        # Wait for remaining tasks
        for future in concurrent.futures.as_completed(pending_futures):
            try:
                count = future.result()
                if count:
                    total_polymers += count
            except Exception as e:
                logging.error(f"Task failed: {e}")
            tasks_completed += 1
            if tasks_completed % 100 == 0:
                logging.info(
                    f"Completed {tasks_completed} tasks. Total polymers so far: {total_polymers}"
                )

    logging.info(f"Processing complete. Total polymers generated: {total_polymers}")


def process_batch_data(
    temp1,
    chunk2_path,
    targ_mon1,
    targ_mon2,
    P_class,
    Ps_rxnL_key,
    targ_rxn,
    monL,
    exclL,
    mon_dic,
    Ps_rxnL,
    output_dir,
    batch_id,
):
    # This helper handles the inner loop logic where temp1 is already loaded

    if targ_mon2 != "none":
        df2 = load_chunk(chunk2_path)
        if targ_mon2 in df2.columns:
            df2[targ_mon2] = (
                df2[targ_mon2].replace({"False": False, "True": True}).astype(bool)
            )

        if targ_mon2 not in df2.columns:
            return 0

        temp2 = df2[df2[targ_mon2]]["smip_cand_mons"].tolist()
        if not temp2:
            return 0

        # Optimization: Process in mini-batches to prevent memory explosion
        MINI_BATCH_SIZE = 5000
        accumulated_dfs = []
        current_batch_data = []

        rxn_smarts = AllChem.ReactionToSmarts(targ_rxn)
        ps_rxn_l_int = int(Ps_rxnL_key)
        p_class_str = str(P_class)

        for m1 in temp1:
            for m2 in temp2:
                if m1 != m2:
                    current_batch_data.append((m1, m2))

                if len(current_batch_data) >= MINI_BATCH_SIZE:
                    df_batch = _process_mini_batch_list(
                        current_batch_data,
                        p_class_str,
                        ps_rxn_l_int,
                        rxn_smarts,
                        targ_rxn,
                        monL,
                        Ps_rxnL,
                    )
                    if not df_batch.empty:
                        accumulated_dfs.append(df_batch)
                    current_batch_data = []

        # Process leftovers
        if current_batch_data:
            df_batch = _process_mini_batch_list(
                current_batch_data,
                p_class_str,
                ps_rxn_l_int,
                rxn_smarts,
                targ_rxn,
                monL,
                Ps_rxnL,
            )
            if not df_batch.empty:
                accumulated_dfs.append(df_batch)

        if not accumulated_dfs:
            return 0

        final_df = pd.concat(accumulated_dfs, ignore_index=True)
        return save_results(final_df, P_class, output_dir, batch_id)


def _process_mini_batch_list(
    data, p_class_str, ps_rxn_l_int, rxn_smarts, targ_rxn, monL, Ps_rxnL
):
    # Process a list of (m1, m2) tuples
    if not data:
        return pd.DataFrame()

    m1s = [x[0] for x in data]
    m2s = [x[1] for x in data]

    polym_results = []
    for m1_smi, m2_smi in data:
        m1 = genmol(m1_smi)
        m2 = genmol(m2_smi)

        # Check for invalid molecules (genmol returns nan on failure)
        # We assume if it's not a Mol object (e.g. float/nan), it failed.
        if not (isinstance(m1, Chem.Mol) and isinstance(m2, Chem.Mol)):
            polym_results.append([])
            continue

        try:
            res = bipolymA(
                [m1, m2],
                targ_rxn=targ_rxn,
                monL=monL,
                Ps_rxnL=Ps_rxnL,
                P_class=p_class_str,
            )
            polym_results.append(res)
        except Exception:
            polym_results.append([])

    df = pd.DataFrame({"mon1": m1s, "mon2": m2s, "polym": polym_results})
    df["polymer_class"] = p_class_str
    df["Ps_rxnL"] = ps_rxn_l_int
    df["Ps_rxn_smarts"] = rxn_smarts
    return df


def process_batch(
    chunk1_path,
    chunk2_path,
    targ_mon1,
    targ_mon2,
    P_class,
    Ps_rxnL_key,
    targ_rxn,
    monL,
    exclL,
    mon_dic,
    Ps_rxnL,
    output_dir,
    batch_id,
):
    # Wrapper for full load
    df1 = load_chunk(chunk1_path)
    if targ_mon1 in df1.columns:
        df1[targ_mon1] = (
            df1[targ_mon1].replace({"False": False, "True": True}).astype(bool)
        )

    if targ_mon1 not in df1.columns:
        return 0

    temp1 = df1[df1[targ_mon1]]["smip_cand_mons"].tolist()
    if not temp1:
        return 0

    if targ_mon2 == "none":
        # Homopolymerization
        temp2 = ["" for _ in range(len(temp1))]
        DF_temp = pd.DataFrame(
            data={"mon1": temp1, "mon2": temp2}, columns=["mon1", "mon2"]
        )
        DF_temp["polymer_class"] = str(P_class)
        DF_temp["Ps_rxnL"] = int(Ps_rxnL_key)
        DF_temp["Ps_rxn_smarts"] = AllChem.ReactionToSmarts(targ_rxn)

        mons = monL[mon_dic[targ_mon1]]
        excls = exclL[mon_dic[targ_mon1]]

        DF_temp["polym"] = DF_temp.apply(lambda x: genmol(x["mon1"]), axis=1).apply(
            homopolymA,
            mons=mons,
            excls=excls,
            targ_mon1=targ_mon1,
            Ps_rxnL=Ps_rxnL,
            mon_dic=mon_dic,
            monL=monL,
        )
        return save_results(DF_temp, P_class, output_dir, batch_id)
    else:
        # Pass to helper
        return process_batch_data(
            temp1,
            chunk2_path,
            targ_mon1,
            targ_mon2,
            P_class,
            Ps_rxnL_key,
            targ_rxn,
            monL,
            exclL,
            mon_dic,
            Ps_rxnL,
            output_dir,
            batch_id,
        )


def save_results(df, P_class, output_dir, batch_id):
    # Common post-processing
    if df.empty:
        return 0

    # Explode polym list if needed (bipolymA returns list of SMILES)
    # homopolymA also returns list of SMILES

    # Logic from biplym:
    # DF_gendP = DF_Pgen.explode("polym")

    df_exploded = df.explode("polym")
    df_exploded = df_exploded.dropna(subset=["polym"])
    df_exploded.replace({"polym": {"": np.nan}}, inplace=True)
    df_exploded = df_exploded.dropna(subset=["polym"])

    if df_exploded.empty:
        return 0

    # Drop duplicates within batch (optional but good)
    if "mon2" in df_exploded.columns:
        df_exploded = df_exploded[df_exploded["mon1"] != df_exploded["mon2"]]

    # We skip the heavy duplicate drop across all data here ("reactset")
    # because we are processing chunks. Deduplication should happen after merging.

    output_file = Path(output_dir) / f"{P_class}_{batch_id}.parquet"
    df_exploded.to_parquet(output_file, index=False)
    # print(f"    Saved results to {output_file.name}")
    return len(df_exploded)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process chunked monomer data for polymerization."
    )
    parser.add_argument(
        "input_dir", help="Directory containing chunked data and metadata.json"
    )
    parser.add_argument("output_dir", help="Directory to save generated polymers")
    parser.add_argument(
        "--targets",
        nargs="+",
        default=["all"],
        help="Specific polymer classes to generate (default: all)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of worker processes (default: CPU count)",
    )

    args = parser.parse_args()

    process_chunks(args.input_dir, args.output_dir, args.targets, args.workers)
