"""
This file is responsible for running benchmarks on various models and datasets.
Each run will be saved as an observation locally. The observation results will be saved as
logs, graphs and tables.
"""

import glob
import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from src.benchmarking.metrics.seq2seq_metrics import Seq2SeqValidityMetrics

logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """
    Runner for executing benchmarks on model outputs.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

    def run(self) -> bool:
        """
        Run the benchmarking process.
        """
        logger.info("Starting benchmarking process...")

        # 1. Resolve Input Data
        input_file = self._resolve_input_file()
        if not input_file:
            return False

        # 2. Resolve Output Directory
        output_dir = self._resolve_output_dir(input_file)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 3. Load Data
        df = self._load_data(input_file)
        if df is None:
            return False

        # 4. Identify Column
        col_name = self._resolve_column_name(df)
        if not col_name:
            return False

        # 5. Run Metrics
        success = True

        # Validity Metric
        if not self._run_validity_metric(df, col_name, output_dir, input_file):
            success = False

        # Add other metrics here in future

        if success:
            logger.info(
                f"Benchmarking completed successfully. Results saved to {output_dir}"
            )
        else:
            logger.warning("Benchmarking completed with errors.")

        return success

    def _resolve_input_file(self) -> Optional[str]:
        input_file = self.config.get("input_file")
        experiment_dir = self.config.get("experiment_dir")

        if input_file:
            if os.path.exists(input_file):
                return input_file
            logger.error(f"Provided input file does not exist: {input_file}")
            return None

        if experiment_dir:
            if not os.path.exists(experiment_dir):
                logger.error(f"Experiment directory does not exist: {experiment_dir}")
                return None

            # Search for result files
            # Priority: defined in config > generated_molecules.csv > samples.csv > any .csv

            # Define search paths including subdirectories
            search_paths = [
                experiment_dir,
                os.path.join(experiment_dir, "results"),
                os.path.join(experiment_dir, "results", "samples"),
            ]

            # Priority filenames
            priority_names = [
                "generated_molecules.csv",
                "samples.csv",
                "generated_molecules.json",
            ]

            # 1. Check for specific priority files in search paths
            for path_dir in search_paths:
                if not os.path.exists(path_dir):
                    continue

                for name in priority_names:
                    file_path = os.path.join(path_dir, name)
                    if os.path.exists(file_path):
                        logger.info(f"Found input file: {file_path}")
                        return file_path

            # 2. Fallback: Search for any CSV using glob in search paths
            for path_dir in search_paths:
                if not os.path.exists(path_dir):
                    continue

                csvs = glob.glob(os.path.join(path_dir, "*.csv"))
                if csvs:
                    # Prefer one with 'generated' or 'sample' in the name
                    candidates = [c for c in csvs if "generated" in c or "sample" in c]
                    chosen = candidates[0] if candidates else csvs[0]
                    logger.info(f"Auto-selected input file: {chosen}")
                    return chosen

            logger.error(
                f"No suitable input file found in experiment directory (checked subdirectories): {experiment_dir}"
            )
            return None

        logger.error("No input_file or experiment_dir provided in configuration.")
        return None

    def _resolve_output_dir(self, input_file: str) -> str:
        output_dir = self.config.get("output_dir")
        if output_dir:
            return output_dir

        experiment_dir = self.config.get("experiment_dir")
        if experiment_dir:
            return os.path.join(experiment_dir, "benchmarks")

        # Default to 'benchmarks' folder next to input file
        return os.path.join(os.path.dirname(input_file), "benchmarks")

    def _load_data(self, filepath: str) -> Optional[pd.DataFrame]:
        try:
            if filepath.endswith(".csv"):
                return pd.read_csv(filepath)
            elif filepath.endswith(".json"):
                return pd.read_json(filepath)
            else:
                # Try CSV by default
                return pd.read_csv(filepath)
        except Exception as e:
            logger.error(f"Failed to load data from {filepath}: {e}")
            return None

    def _resolve_column_name(self, df: pd.DataFrame) -> Optional[str]:
        # Configured column
        col = self.config.get("csv_column")
        if col and col in df.columns:
            return col

        if col:
            logger.warning(f"Configured column '{col}' not found in data.")

        # Common defaults
        defaults = ["sequence", "SMILES", "smiles", "generated_sequence", "molecule"]
        for c in defaults:
            if c in df.columns:
                logger.info(f"Using column '{c}' for benchmarking.")
                return c

        logger.error(
            f"Could not identify sequence column. Available columns: {df.columns.tolist()}"
        )
        return None

    def _run_validity_metric(
        self, df: pd.DataFrame, col_name: str, output_dir: str, input_file: str
    ) -> bool:
        try:
            logger.info("Running Validity Benchmark...")
            metrics = Seq2SeqValidityMetrics(df, col_name)
            score, details = metrics.compute()

            logger.info(f"Validity Score: {score:.4f}")

            # Save results as JSON
            results_json_path = os.path.join(output_dir, "validity_results.json")
            json_results = {
                "date": str(pd.Timestamp.now()),
                "input_file": input_file,
                "column": col_name,
                "total_sequences": len(df),
                "validity_score": score,
                "error_details": details,
            }
            with open(results_json_path, "w") as f:
                json.dump(json_results, f, indent=4)

            # Save textual results
            results_path = os.path.join(output_dir, "validity_results.txt")
            with open(results_path, "w") as f:
                f.write("=== Validity Benchmark Results ===\n")
                f.write(f"Date: {pd.Timestamp.now()}\n")
                f.write(f"Input File: {input_file}\n")
                f.write(f"Column: {col_name}\n")
                f.write(f"Total Sequences: {len(df)}\n")
                f.write("-" * 30 + "\n")
                f.write(f"Validity Score: {score:.6f}\n")
                f.write("-" * 30 + "\n")
                f.write("Error Details:\n")
                for k, v in details.items():
                    f.write(f"  {k}: {v:.6f}\n")

            # Save visualization
            plot_path = os.path.join(output_dir, "validity_distribution.png")
            metrics.visualize(save_path=plot_path)

            return True
        except Exception as e:
            logger.error(f"Error running validity metric: {e}")
            # import traceback
            # traceback.print_exc()
            return False


def run_benchmark(config: Dict[str, Any]) -> bool:
    """
    Entry point to run benchmarks with a given configuration.
    """
    runner = BenchmarkRunner(config)
    return runner.run()
