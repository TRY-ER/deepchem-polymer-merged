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

from smipoly.batch_scripts.chunk_data import chunk_data


def main(input_csv,
         output_dir,
         chunk_dir,
         results_dir,
         chunk_size = 1000):
    print(f"Running demo with input: {input_csv}")

    if not input_csv.exists():
        print(f"Error: Input file not found at {input_csv}")
        return

    # Clean up previous run
    if output_dir.exists():
        print(f"Cleaning up previous demo directory: {output_dir}")
        shutil.rmtree(output_dir)

    chunk_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # 1. Chunk the data
    # We use a small chunk size to demonstrate how it splits the file.
    # The sample file is small, so we use 100 rows per chunk.
    # In production, use 1000-10000 depending on memory.
    print("\nStep 1: Chunking data...")
    chunk_data(
        input_file=str(input_csv),
        output_dir=str(chunk_dir),
        chunk_size=chunk_size,
        file_format="parquet",
    )

if __name__ == "__main__":
    # Configuration
    # We use the sample data provided in the repository
    input_csv = smipoly_root / "monomer_classified_df.csv"

    # Create a demo directory for output
    output_dir = smipoly_root / "output"
    chunk_dir = output_dir / "chunks"
    results_dir = output_dir / "results"
   
    CHUNK_SIZE = 1000

    main(input_csv,
         output_dir,
         chunk_dir,
         results_dir,
         chunk_size = CHUNK_SIZE)
