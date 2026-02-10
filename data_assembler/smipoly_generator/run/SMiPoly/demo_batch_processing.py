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
from smipoly.batch_scripts.process_chunks import process_chunks


def main():
    # Configuration
    # We use the sample data provided in the repository
    input_csv = smipoly_root / "monomer_classified_df.csv"

    # Create a demo directory for output
    demo_dir = smipoly_root / "demo_output"
    chunk_dir = demo_dir / "chunks"
    results_dir = demo_dir / "results"

    print(f"Running demo with input: {input_csv}")

    if not input_csv.exists():
        print(f"Error: Input file not found at {input_csv}")
        return

    # Clean up previous run
    if demo_dir.exists():
        print(f"Cleaning up previous demo directory: {demo_dir}")
        shutil.rmtree(demo_dir)

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
        chunk_size=100,
        file_format="parquet",
    )

    # 2. Process the chunks
    print("\nStep 2: Processing chunks...")
    # This reads the metadata from the chunk directory and processes relevant pairs.
    # We target "all" classes to see what it can generate from the sample data.
    process_chunks(
        input_dir=str(chunk_dir),
        output_dir=str(results_dir),
        specific_targets=["all"],
        max_workers=4,
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
    main()
