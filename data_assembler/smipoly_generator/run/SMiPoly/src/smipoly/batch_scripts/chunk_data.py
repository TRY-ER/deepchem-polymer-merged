import argparse
import json
from pathlib import Path

import pandas as pd


def chunk_data(input_file, output_dir, chunk_size=1000, file_format="parquet"):
    """
    Reads a large input file (CSV or Parquet) in chunks and saves them as separate files in the output directory.

    Args:
        input_file (str): Path to the input file.
        output_dir (str): Path to the directory where chunks will be saved.
        chunk_size (int): Number of rows per chunk.
        file_format (str): Output format ('parquet' or 'csv').
    """
    input_path = Path(input_file)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Processing {input_file}...")
    print(f"Output directory: {output_dir}")
    print(f"Chunk size: {chunk_size}")

    # Create a metadata file to store info about the batches
    metadata = {
        "original_file": str(input_path),
        "chunk_size": chunk_size,
        "total_rows": 0,
        "num_chunks": 0,
        "files": [],
    }

    try:
        # Determine file type and read accordingly
        if input_path.suffix == ".csv":
            # Use pandas read_csv with chunksize
            reader = pd.read_csv(input_path, chunksize=chunk_size)

            total_rows = 0
            chunk_idx = 0

            for chunk in reader:
                chunk_filename = f"batch_{chunk_idx}.{file_format}"
                chunk_file_path = output_path / chunk_filename

                if file_format == "parquet":
                    chunk.to_parquet(chunk_file_path, index=False)
                else:
                    chunk.to_csv(chunk_file_path, index=False)

                rows_in_chunk = len(chunk)
                total_rows += rows_in_chunk

                print(
                    f"Saved chunk {chunk_idx} with {rows_in_chunk} rows to {chunk_filename}"
                )

                metadata["files"].append(
                    {
                        "filename": chunk_filename,
                        "rows": rows_in_chunk,
                        "index": chunk_idx,
                    }
                )

                chunk_idx += 1

            metadata["total_rows"] = total_rows
            metadata["num_chunks"] = chunk_idx

        elif input_path.suffix == ".parquet":
            import pyarrow.parquet as pq

            parquet_file = pq.ParquetFile(input_path)

            total_rows = 0
            chunk_idx = 0

            for batch in parquet_file.iter_batches(batch_size=chunk_size):
                chunk = batch.to_pandas()
                chunk_filename = f"batch_{chunk_idx}.{file_format}"
                chunk_file_path = output_path / chunk_filename

                if file_format == "parquet":
                    chunk.to_parquet(chunk_file_path, index=False)
                else:
                    chunk.to_csv(chunk_file_path, index=False)

                rows_in_chunk = len(chunk)
                total_rows += rows_in_chunk

                print(
                    f"Saved chunk {chunk_idx} with {rows_in_chunk} rows to {chunk_filename}"
                )

                metadata["files"].append(
                    {
                        "filename": chunk_filename,
                        "rows": rows_in_chunk,
                        "index": chunk_idx,
                    }
                )

                chunk_idx += 1

            metadata["total_rows"] = total_rows
            metadata["num_chunks"] = chunk_idx

        else:
            raise ValueError(f"Unsupported input file format: {input_path.suffix}")

        # Save metadata
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=4)

        print("Chunking complete.")

    except Exception as e:
        print(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Chunk large dataset into smaller files."
    )
    parser.add_argument("input_file", help="Path to input file (CSV or Parquet)")
    parser.add_argument("output_dir", help="Directory to save output chunks")
    parser.add_argument(
        "--chunk_size", type=int, default=1000, help="Number of rows per chunk"
    )
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        help="Output file format",
    )

    args = parser.parse_args()

    chunk_data(args.input_file, args.output_dir, args.chunk_size, args.format)
