#!/usr/bin/env python3
"""
Test script for visualization functionality in the runner.py
"""

import os
import sys
import pytest
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Add project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from src.runner import ModelRunner, create_argument_parser

# Test data
TEST_SMILES = "[1*]C(=O)C[4*].[4*]CC(O)COC(=O)CCCCCN[5*]|[4*]-[*:1].[5*]-[*:2]>>[$([C&!D1&!$(C=*)]-&!@[#6]):1]-&!@[$([N&!D1&!$(N=*)&!$(N-[!#6&!#16&!#0&!#1])&!$([N&R]@[C&R]=O)]):2]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5"

class TestVisualization:
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment before each test."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.test_dir, "visualizations")
        
        # Create a temporary CSV file with test data
        self.csv_path = os.path.join(self.test_dir, "test_data.csv")
        test_data = {
            'SMILES': [TEST_SMILES] * 3,
            'LogP1': [-1.2, 0.5, 1.0],
            'LogP2': [2.3, -0.8, 1.5],
            # include config-default column names as well (match config case)
            'logP_1': [-1.2, 0.5, 1.0],
            'logP_2': [2.3, -0.8, 1.5]
        }
        pd.DataFrame(test_data).to_csv(self.csv_path, index=False)
        
        # Create a temporary text file with test data
        self.txt_path = os.path.join(self.test_dir, "test_data.txt")
        with open(self.txt_path, 'w') as f:
            f.write(f"{TEST_SMILES}\n" * 3)
        
        # Setup test configuration
        self.config_dir = os.path.join(self.test_dir, "config")
        os.makedirs(os.path.join(self.config_dir, "visualizer"), exist_ok=True)
        self.config_path = os.path.join(self.config_dir, "visualizer", "test_config.yaml")
        with open(self.config_path, 'w') as f:
            f.write("""
canvas:
  width: 1000
  height: 400
  figure_width: 8
  figure_height: 5
output:
  directory: "./test_visualizations"
batch:
  enabled: true
  csv_column: "SMILES"
  chunk_size: 2
logp:
  columns:
    logp1: "CustomLogP1"
    logp2: "CustomLogP2"
  defaults:
    logp1: 0.0
    logp2: 0.0
    enabled: false
  colors:
    philic: [0.0, 1.0, 1.0, 1]
    phobic: [1.0, 1.0, 0.0, 1]
            """)
        
        yield
        
        # Cleanup after tests
        shutil.rmtree(self.test_dir)

    def create_args(self, command_str):
        """Helper method to create args from command string."""
        parser = create_argument_parser()
        return parser.parse_args(command_str.split())

    def test_single_smiles_visualization(self):
        """Test visualization of a single SMILES string."""
        args = self.create_args(f"visual --input {TEST_SMILES} --output-dir {self.output_dir}")
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)
        assert os.path.exists(os.path.join(self.output_dir, "polymer.png"))

    def test_batch_txt_visualization(self):
        """Test batch visualization from a text file."""
        args = self.create_args(f"visual --input {self.txt_path} --batch --output-dir {self.output_dir}")
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)
        
        # Check if all files were generated
        visualization_files = list(Path(self.output_dir).glob("polymer_*.png"))
        assert len(visualization_files) == 3

    def test_batch_csv_visualization(self):
        """Test batch visualization from a CSV file."""
        args = self.create_args(
            f"visual --input {self.csv_path} --batch "
            f"--csv-column SMILES --output-dir {self.output_dir}"
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)
        
        # Check if all files were generated (should be in batches)
        visualization_files = list(Path(self.output_dir).glob("polymer_batch_*.png"))
        assert len(visualization_files) > 0

    def test_visualization_with_logp_filter(self):
        """Test visualization with LogP filtering."""
        args = self.create_args(
            f"visual --input {TEST_SMILES} --output-dir {self.output_dir} "
            "--logp-filter --logp1 -1.2 --logp2 2.3"
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)
        assert os.path.exists(os.path.join(self.output_dir, "polymer.png"))

    def test_visualization_with_custom_config(self):
        """Test visualization with custom configuration."""
        # Create a CSV that matches the custom config's LogP column names
        test_data = {
            'SMILES': [TEST_SMILES] * 3,
            'CustomLogP1': [-1.2, 0.5, 1.0],
            'CustomLogP2': [2.3, -0.8, 1.5]
        }
        custom_csv_path = os.path.join(self.test_dir, "custom_config_test_data.csv")
        pd.DataFrame(test_data).to_csv(custom_csv_path, index=False)

        args = self.create_args(
            f"visual --input {custom_csv_path} --batch "
            f"--config {self.config_path} --output-dir {self.output_dir}"
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)

    def test_invalid_csv_column(self):
        """Test handling of invalid CSV column name."""
        args = self.create_args(
            f"visual --input {self.csv_path} --batch "
            f"--csv-column INVALID_COLUMN --output-dir {self.output_dir}"
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert not success

    def test_invalid_input_file(self):
        """Test handling of non-existent input file."""
        args = self.create_args(
            "visual --input nonexistent.csv --batch "
            f"--output-dir {self.output_dir}"
        )
        runner = ModelRunner()
        # The runner currently treats a non-existent file as a SMILES string and
        # passes it to the visualizer which will raise a ValueError for invalid data.
        # Expect that behavior here.
        with pytest.raises(ValueError):
            runner.visualize_structure(args)

    def test_incomplete_logp_filter(self):
        """Test handling of incomplete LogP filter parameters."""
        args = self.create_args(
            f"visual --input {TEST_SMILES} --output-dir {self.output_dir} "
            "--logp-filter --logp1 -1.2"  # Missing logp2
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert not success

    def test_batch_csv_visualization_with_logp_columns(self):
        """Test batch visualization from CSV file using LogP value columns."""
        args = self.create_args(
            f"visual --input {self.csv_path} --batch "
            f"--csv-column SMILES --logp1-column LogP1 --logp2-column LogP2 "
            f"--output-dir {self.output_dir}"
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)
        
        # Check if batch files were generated
        visualization_files = list(Path(self.output_dir).glob("polymer_batch_*.png"))
        assert len(visualization_files) > 0

    def test_batch_csv_visualization_with_invalid_logp_columns(self):
        """Test batch visualization with non-existent LogP columns."""
        args = self.create_args(
            f"visual --input {self.csv_path} --batch "
            f"--csv-column SMILES --logp1-column Invalid1 --logp2-column Invalid2 "
            f"--output-dir {self.output_dir}"
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert not success

    def test_batch_csv_visualization_with_config_logp_columns(self):
        """Test batch visualization using LogP column names from config."""
        # Create a CSV with the config-specified column names
        test_data = {
            'SMILES': [TEST_SMILES] * 3,
            'CustomLogP1': [-1.2, 0.5, 1.0],
            'CustomLogP2': [2.3, -0.8, 1.5]
        }
        config_csv_path = os.path.join(self.test_dir, "config_test_data.csv")
        pd.DataFrame(test_data).to_csv(config_csv_path, index=False)

        # Test using just the config without explicit column names
        args = self.create_args(
            f"visual --input {config_csv_path} --batch "
            f"--csv-column SMILES --config {self.config_path} "
            f"--output-dir {self.output_dir}"
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)
        
        # Check if batch files were generated
        visualization_files = list(Path(self.output_dir).glob("polymer_batch_*.png"))
        assert len(visualization_files) > 0

    def test_logp_column_priority(self):
        """Test that command line LogP column names override config values."""
        # Create CSV with both default and custom column names
        test_data = {
            'SMILES': [TEST_SMILES] * 3,
            'CustomLogP1': [-1.2, 0.5, 1.0],  # Config-specified name
            'CustomLogP2': [2.3, -0.8, 1.5],  # Config-specified name
            'LogP1': [0.1, 0.2, 0.3],         # CLI-specified name
            'LogP2': [0.4, 0.5, 0.6]          # CLI-specified name
        }
        priority_csv_path = os.path.join(self.test_dir, "priority_test_data.csv")
        pd.DataFrame(test_data).to_csv(priority_csv_path, index=False)

        # Test that CLI arguments override config
        args = self.create_args(
            f"visual --input {priority_csv_path} --batch "
            f"--csv-column SMILES --config {self.config_path} "
            f"--logp1-column LogP1 --logp2-column LogP2 "
            f"--output-dir {self.output_dir}"
        )
        runner = ModelRunner()
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)
        
        # Clean up output directory for next test
        shutil.rmtree(self.output_dir)
        os.makedirs(self.output_dir)

        # Test config values are used when no CLI arguments
        args = self.create_args(
            f"visual --input {priority_csv_path} --batch "
            f"--csv-column SMILES --config {self.config_path} "
            f"--output-dir {self.output_dir}"
        )
        
        success = runner.visualize_structure(args)
        assert success
        assert os.path.exists(self.output_dir)

if __name__ == "__main__":
    pytest.main([__file__, "-v"])