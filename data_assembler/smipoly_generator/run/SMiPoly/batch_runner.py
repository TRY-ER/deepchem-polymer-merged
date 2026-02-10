import shutil
import sys
from pathlib import Path

import pandas as pd

# Add src to path to allow importing the batch scripts
script_path = Path(__file__).resolve()
smipoly_root = script_path.parent
src_path = smipoly_root / "src"

if str(src_path) not in sys.path:
    sys.path.append(str(src_path))

from smipoly.batch_scripts.process_chunks import process_chunks


def main(chunk_dir,
         results_dir,
         max_workers = 4):
   # 1. Process the chunks
    print("\nStep 1: Processing chunks...")
    # This reads the metadata from the chunk directory and processes relevant pairs.
    # We target "all" classes to see what it can generate from the sample data.
    process_chunks(
        input_dir=str(chunk_dir),
        output_dir=str(results_dir),
        specific_targets=["all"],
        max_workers=22,
    )

    print("\nProcessing complete!")
    print(f"Results are saved in: {results_dir}")

    # Inspect Results
    results = list(results_dir.glob("*.parquet"))
    print(f"Generated {len(results)} output files.")

    if results:
        print("\nSample of generated data:")
        # Read the first result file to show some output
        first_file = results[0]
        print(f"Reading {first_file.name}...")
        df = pd.read_parquet(first_file)
        print(df[["mon1", "mon2", "polym", "polymer_class"]].head())
    else:
        print(
            "No polymers generated from this sample data (likely no matching reactive pairs found)."
        )


if __name__ == "__main__":
    output_dir = smipoly_root / "output"
    chunk_dir = output_dir / "chunks"
    results_dir = output_dir / "results"

    MAX_WORKERS = 22

    main(
        chunk_dir = chunk_dir,
        results_dir = results_dir,
        max_workers = MAX_WORKERS
    )
