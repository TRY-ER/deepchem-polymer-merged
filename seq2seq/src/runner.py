#!/usr/bin/env python3
"""
Comprehensive Runner Script for Polymer Training, Sampling and Visualization
==========================================================================

This script provides a unified interface for:
1. Training models with structured configurations
2. Sampling from trained models with optimized parameters
3. Managing experiments with proper directory organization
4. Visualizing polymer structures with configurable options

Usage:
    # Training
    python runner.py train --model gru --config config/trainer/gru_config.yaml
    python runner.py train --model transformer --epochs 50 --batch_size 64

    # Sampling
    python runner.py sample --experiment experiments/gru_20240806_123456
    python runner.py sample --checkpoint checkpoints/best_model.pth --model transformer

    # List experiments
    python runner.py list

    # Show config
    python runner.py show-config --model gru --mode train

    # Visualization
    # Single SMILES string
    python runner.py visual --input "SMILES_STRING" --output-dir visualizations

    # Batch from CSV (basic)
    python runner.py visual --input polymers.csv --batch --csv-column SMILES --output-dir visualizations

    # Batch from CSV with LogP columns (reads LogP values for each molecule)
    python runner.py visual --input data.csv --batch --csv-column SMILES \
                          --logp1-column LogP1 --logp2-column LogP2

    # Text file with one SMILES per line
    python runner.py visual --input polymers.txt --batch --output-dir visualizations

    # Global LogP filtering (same values for all molecules)
    python runner.py visual --input "SMILES_STRING" --logp-filter --logp1 -1.2 --logp2 2.3
    
    # Default LogP columns from config
    # Uses column names defined in config/visualizer/base_config.yaml
    python runner.py visual --input data.csv --batch --csv-column SMILES

    # Batch visualization with LogP values from CSV columns (explicit column names)
    python runner.py visual --input data.csv --batch --csv-column SMILES --logp1-column LogP1 --logp2-column LogP2

    # Batch visualization using LogP column names from config
    # (config/visualizer/base_config.yaml defines the column names under logp.columns)
    python runner.py visual --input data.csv --batch --csv-column SMILES

Notes:
    - Visualization configuration lives under `config/visualizer/base_config.yaml`.
    - CSV batch processing uses pandas (must be installed in your environment).
    - The visualizer produces PNG files by default and accepts CLI overrides for canvas
      and figure sizes as well as output directory.
"""

import argparse
import os
import sys
import yaml
import json
import glob
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir) 
sys.path.insert(0, project_root)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ConfigManager:
    """Manages configuration loading and merging for training and sampling."""
    
    def __init__(self, config_root: str = "config"):
        self.config_root = config_root
        self.trainer_config_dir = os.path.join(config_root, "trainer")
        self.sampler_config_dir = os.path.join(config_root, "sampler")
    
    def load_yaml(self, filepath: str) -> Dict[str, Any]:
        """Load YAML configuration file."""
        try:
            with open(filepath, 'r') as f:
                return yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"Config file not found: {filepath}")
            return {}
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {filepath}: {e}")
            return {}
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries."""
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def load_trainer_config(self, model_type: str, custom_config: Optional[str] = None) -> Dict[str, Any]:
        """Load trainer configuration for a specific model type."""
        # Load base configuration
        base_config_path = os.path.join(self.trainer_config_dir, "base_config.yaml")
        base_config = self.load_yaml(base_config_path)
        
        # Load model-specific configuration
        model_config_path = os.path.join(self.trainer_config_dir, f"{model_type}_config.yaml")
        model_config = self.load_yaml(model_config_path)
        
        # Merge configurations
        config = self.merge_configs(base_config, model_config)
        
        # Load custom configuration if provided
        if custom_config and os.path.exists(custom_config):
            custom_config_data = self.load_yaml(custom_config)
            config = self.merge_configs(config, custom_config_data)
        
        # Ensure model type is set
        if 'model' not in config:
            config['model'] = {}
        config['model']['type'] = model_type
        
        return config
    
    def load_sampler_config(self, model_type: str, custom_config: Optional[str] = None) -> Dict[str, Any]:
        """Load sampler configuration for a specific model type."""
        # Load base configuration
        base_config_path = os.path.join(self.sampler_config_dir, "base_config.yaml")
        base_config = self.load_yaml(base_config_path)
        
        # Load model-specific configuration
        model_config_path = os.path.join(self.sampler_config_dir, f"{model_type}_config.yaml")
        model_config = self.load_yaml(model_config_path)
        
        # Merge configurations
        config = self.merge_configs(base_config, model_config)
        
        # Load custom configuration if provided
        if custom_config and os.path.exists(custom_config):
            custom_config_data = self.load_yaml(custom_config)
            config = self.merge_configs(config, custom_config_data)
        
        return config


class ExperimentManager:
    """Manages experiment discovery and information."""
    
    def __init__(self, experiments_dir: str = "experiments"):
        self.experiments_dir = experiments_dir
    
    def list_experiments(self) -> List[Dict[str, Any]]:
        """List all available experiments."""
        if not os.path.exists(self.experiments_dir):
            return []
        
        experiments = []
        for exp_dir in os.listdir(self.experiments_dir):
            exp_path = os.path.join(self.experiments_dir, exp_dir)
            if os.path.isdir(exp_path):
                exp_info = self._get_experiment_info(exp_path)
                experiments.append(exp_info)
        
        # Sort by creation time (newest first)
        experiments.sort(key=lambda x: x['created'], reverse=True)
        return experiments
    
    def _get_experiment_info(self, exp_path: str) -> Dict[str, Any]:
        """Get information about a specific experiment."""
        exp_name = os.path.basename(exp_path)
        
        # Try to load experiment info from config
        config_path = os.path.join(exp_path, "config", "model_config.json")
        model_type = "unknown"
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    model_type = config.get('model_type', 'unknown')
            except:
                pass
        
        # Get creation time
        created = datetime.fromtimestamp(os.path.getctime(exp_path))
        
        # Check for checkpoints
        checkpoints_dir = os.path.join(exp_path, "checkpoints")
        has_best = os.path.exists(os.path.join(checkpoints_dir, "best_model.pth"))
        has_final = os.path.exists(os.path.join(checkpoints_dir, "final_model.pth"))
        
        # Check for training history
        history_path = os.path.join(exp_path, "results", "training_history.json")
        training_completed = os.path.exists(history_path)
        
        return {
            'name': exp_name,
            'path': exp_path,
            'model_type': model_type,
            'created': created,
            'has_best_checkpoint': has_best,
            'has_final_checkpoint': has_final,
            'training_completed': training_completed
        }


class ModelRunner:
    """Main runner class that coordinates training and sampling."""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.experiment_manager = ExperimentManager()
        # directory for visualizer configs
        self.visualizer_config_dir = os.path.join("config", "visualizer")
    
    def train_model(self, args):
        """Train a model with the specified configuration."""
        logger.info(f"🚀 Starting training for {args.model} model")
        
        # Load configuration
        config = self.config_manager.load_trainer_config(args.model, args.config)

        
        # Override config with command line arguments
        self._override_trainer_config(config, args)
        

        # Import trainer
        try:
            from src.trainer.trainer import create_trainer
        except ImportError as e:
            logger.error(f"Failed to import trainer: {e}")
            return False
        
        # Create trainer with configuration
        trainer_kwargs = self._build_trainer_kwargs(config)

        data_config_kwargs = {}

        print("config >>", config)

        if "data" in config: 
            if "data_config" in config["data"]:
                print("this is getting triggered !")
                data_config_kwargs = {"data_config": config["data"]["data_config"]} 

        penalty_config = config.get("penalty_config", {})
        logging_config = config.get("logging", {}) 
        
        if "penalty_config" in config:
            penalty_config = {"penalty_config": config["penalty_config"]}

        try:
            trainer = create_trainer(
                model_name=args.model,
                vocab_size=config['model'].get('vocab_size', 28996),
                **trainer_kwargs,
                **data_config_kwargs,
                **penalty_config,
                **logging_config
            )
            
            logger.info(f"📁 Experiment directory: {trainer.experiment_dir}")
            
            # Train the model
            history = trainer.train()
            
            logger.info("✅ Training completed successfully!")
            logger.info(f"📊 Final test loss: {history.get('test_loss', 'N/A'):.4f}")
            logger.info(f"📁 Results saved to: {trainer.experiment_dir}")
            
            return True
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return False
    
    def sample_from_model(self, args):
        """Sample from a trained model."""
        logger.info(f"🧪 Starting sampling")
        
        # Determine checkpoint path and model type
        if args.experiment:
            experiment_dir = args.experiment
            # Extract model type from experiment directory name
            exp_name = os.path.basename(experiment_dir.rstrip('/'))
            model_type = exp_name.split('_')[0] if '_' in exp_name else args.model
            
            # Load sampling configuration
            config = self.config_manager.load_sampler_config(model_type, args.config)
            
            # Set experiment configuration
            experiment_config = config.setdefault('experiment', {})
            experiment_config['experiment_dir'] = experiment_dir
            
            # Override config with command line arguments
            self._override_sampler_config(config, args)
            
            # Use experiment directory as checkpoint path for sampler
            checkpoint_path = experiment_dir
            
        else:
            checkpoint_path = args.checkpoint
            model_type = args.model
            
            if not model_type:
                logger.error("Model type must be specified when using direct checkpoint path")
                return False
            
            # Load sampling configuration
            config = self.config_manager.load_sampler_config(model_type, args.config)
            
            # Override config with command line arguments
            self._override_sampler_config(config, args)
        
        # Import sampler
        try:
            from src.sampler.sampler import create_sampler
        except ImportError as e:
            logger.error(f"Failed to import sampler: {e}")
            return False
        
        # print("kwargs >>", config)
        # Create sampler
        sampler_kwargs = self._build_sampler_kwargs(config)
        filter_kwargs = self._build_filter_kwargs(config)

        # print("filter kwargs >>", filter_kwargs)
        
        try:
            sampler = create_sampler(
                checkpoint_path=checkpoint_path,
                sampling_kwargs=sampler_kwargs,
                filter_kwargs=filter_kwargs
            )
            
            logger.info(f"🤖 Model loaded: {sampler.model_type}")
            if sampler.experiment_dir:
                logger.info(f"📁 Experiment directory: {sampler.experiment_dir}")
            
            # Generate samples
            batch_config = config.get('batch', {})
            results = sampler.generate_batch(
                num_sequences=batch_config.get('num_sequences', 10),
                max_length=config['generation'].get('max_length', 100)
            )
            
            # Evaluate results
            if config.get('evaluation', {}).get('calculate_metrics', True):
                evaluation = sampler.evaluate_sequences(results)
                logger.info(f"📊 Generated {evaluation['num_sequences']} sequences")
                logger.info(f"📏 Average length: {evaluation['avg_length']:.1f}")
                logger.info(f"🎯 Diversity ratio: {evaluation['diversity_ratio']:.3f}")
            
            # Save results
            output_config = config.get('output', {})
            formats = output_config.get('formats', ['json'])
            base_filename = output_config.get('base_filename', 'generated_molecules')
            
            saved_files = []
            for fmt in formats:
                filename = f"{base_filename}.{fmt}"
                output_path = sampler.save_results(
                    results, 
                    filename, 
                    format=fmt,
                    use_experiment_dir=output_config.get('use_experiment_dir', True)
                )
                saved_files.append(output_path)
            
            logger.info("✅ Sampling completed successfully!")
            logger.info(f"💾 Results saved to: {len(saved_files)} files")
            
            return True
            
        except Exception as e:
            logger.error(f"Sampling failed: {e}")
            return False
    
    def list_experiments(self, args):
        """List all available experiments."""
        experiments = self.experiment_manager.list_experiments()
        
        if not experiments:
            logger.info("No experiments found in the experiments directory.")
            return
        
        logger.info(f"📁 Found {len(experiments)} experiments:")
        logger.info("-" * 80)
        
        for exp in experiments:
            status_icons = []
            if exp['training_completed']:
                status_icons.append("✅")
            if exp['has_best_checkpoint']:
                status_icons.append("🏆")
            if exp['has_final_checkpoint']:
                status_icons.append("💾")
            
            status = " ".join(status_icons) if status_icons else "⚠️"
            
            logger.info(f"{status} {exp['name']}")
            logger.info(f"    Model: {exp['model_type']}")
            logger.info(f"    Created: {exp['created'].strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"    Path: {exp['path']}")
            logger.info("")
    
    def show_config(self, args):
        """Show configuration for a model and mode."""
        if args.mode == 'train':
            config = self.config_manager.load_trainer_config(args.model)
        elif args.mode == 'sample':
            config = self.config_manager.load_sampler_config(args.model)
        else:
            logger.error("Mode must be 'train' or 'sample'")
            return
        
        logger.info(f"📋 Configuration for {args.model} ({args.mode} mode):")
        logger.info("-" * 50)
        print(yaml.dump(config, default_flow_style=False, indent=2))

    def load_visualizer_config(self, custom_config: Optional[str] = None) -> Dict[str, Any]:
        """Load visualizer configuration from config/visualizer.

        If custom_config is provided and exists, it will be merged on top of the base config.
        """
        base_path = os.path.join(self.visualizer_config_dir, "base_config.yaml")
        base = self.config_manager.load_yaml(base_path) or {}
        if custom_config and os.path.exists(custom_config):
            custom = self.config_manager.load_yaml(custom_config) or {}
            return self.config_manager.merge_configs(base, custom)
        return base

    def visualize_structure(self, args) -> bool:
        """Visualize polymer structures with LogP-based coloring support.

        This method provides flexible visualization options for polymer structures, including
        single SMILES strings and batch processing with LogP-based fragment coloring.

        Features:
        - Single SMILES string visualization
        - Batch processing from text files (one SMILES per line)
        - CSV batch processing with configurable column names
        - LogP-based fragment coloring with two modes:
          1. Global LogP values for all structures
          2. Per-molecule LogP values from CSV columns

        The LogP visualization can be configured in three ways:
        1. Command line arguments (highest priority):
           --logp-filter --logp1 VALUE --logp2 VALUE
           --logp1-column NAME --logp2-column NAME

        2. Configuration file (middle priority):
           logp:
             columns:
               logp1: "LogP1"
               logp2: "LogP2"
             defaults:
               logp1: 0.0
               logp2: 0.0
               enabled: false

        3. Default values (lowest priority):
           Uses "LogP1" and "LogP2" as column names if not specified

        Args:
            args: Command line arguments containing visualization parameters

        Returns:
            bool: True if visualization was successful, False otherwise

        Raises:
            ImportError: If visualization dependencies are not available
            ValueError: If invalid configuration is provided
        """
        try:
            from src.visualizer.base import CustomVisualizer
        except Exception as e:
            logger.error(f"Failed to import visualizer: {e}")
            return False

        # Load config
        config = self.load_visualizer_config(getattr(args, 'config', None)) or {}

        # Resolve canvas/figure sizes and output dir (CLI overrides config)
        canvas_cfg = config.get('canvas', {})
        canvas_width = args.canvas_width if getattr(args, 'canvas_width', None) is not None else canvas_cfg.get('width', 1400)
        canvas_height = args.canvas_height if getattr(args, 'canvas_height', None) is not None else canvas_cfg.get('height', 500)
        fig_cfg = canvas_cfg
        figure_width = args.figure_width if getattr(args, 'figure_width', None) is not None else fig_cfg.get('figure_width', 10)
        figure_height = args.figure_height if getattr(args, 'figure_height', None) is not None else fig_cfg.get('figure_height', 6)
        output_dir = args.output_dir if getattr(args, 'output_dir', None) else config.get('output', {}).get('directory', './visualizations')

        # Configure LogP coloring
        logp_config = config.get('logp', {})
        logp_colors = logp_config.get('colors', {
            'philic': [0.0, 1.0, 1.0, 1],
            'phobic': [1.0, 1.0, 0.0, 1]
        })

        # Initialize visualizer with base configuration
        visualizer = CustomVisualizer(canvas_size=(canvas_width, canvas_height),
                                    figure_size=(figure_width, figure_height),
                                    output_dir=output_dir,
                                    logP_color=None)  # We'll apply LogP coloring through the filter

        # Prepare LogP filter based on priority order
        logP_filter = None
        
        # 1. Command-line LogP filter (highest priority)
        if getattr(args, 'logp_filter', False):
            if args.logp1 is None or args.logp2 is None:
                logger.error("--logp-filter requires --logp1 and --logp2")
                return False
            logP_filter = {"logP_1": args.logp1, "logP_2": args.logp2}
        # Check for default LogP filter in config
        elif logp_config.get('defaults', {}).get('enabled', False):
            logP_filter = {
                "logP_1": logp_config['defaults'].get('logp1', 0.0),
                "logP_2": logp_config['defaults'].get('logp2', 0.0)
            }

        # Determine input type
        input_path = args.input
        # try:
        if os.path.isfile(input_path):
            ext = os.path.splitext(input_path)[1].lower()
            if getattr(args, 'batch', False):
                # Batch mode
                if ext == '.csv':
                    try:
                        import pandas as pd
                    except Exception as e:
                        logger.error(f"Pandas is required to read CSV files: {e}")
                        return False
                    csv_col = getattr(args, 'csv_column', None) or config.get('batch', {}).get('csv_column', 'SMILES')
                    logp_columns = config.get('logp', {}).get('columns', {})
                    logp1_col = getattr(args, 'logp1_column', None) or logp_columns.get('logp1', 'LogP1')
                    logp2_col = getattr(args, 'logp2_column', None) or logp_columns.get('logp2', 'LogP2')
                    chunk_size = config.get('batch', {}).get('chunk_size', 100)
                    
                    # Log which columns will be used for LogP values
                    if logp1_col and logp2_col:
                        source = "command line" if getattr(args, 'logp1_column', None) else "config"
                        logger.info(f"Using LogP columns from {source}:")
                        logger.info(f"  LogP1 column: {logp1_col}")
                        logger.info(f"  LogP2 column: {logp2_col}")
                    
                    for i, chunk in enumerate(pd.read_csv(input_path, chunksize=chunk_size)):
                        if csv_col not in chunk.columns:
                            logger.error(f"CSV column '{csv_col}' not found")
                            return False
                        
                        data_list = chunk[csv_col].dropna().astype(str).tolist()
                        
                        # Handle LogP values with priority:
                        # 1. Column-based LogP values
                        # 2. Global LogP filter
                        # 3. No LogP filtering
                        logp_filters = None
                        if logp1_col and logp2_col:
                            if logp1_col not in chunk.columns or logp2_col not in chunk.columns:
                                logger.error(f"LogP columns not found in CSV: expected '{logp1_col}' and '{logp2_col}'")
                                return False
                            
                            try:
                                logp_filters = []
                                for idx in range(len(data_list)):
                                    logp_filters.append({
                                        "logP_1": float(chunk[logp1_col].iloc[idx]),
                                        "logP_2": float(chunk[logp2_col].iloc[idx])
                                    })
                                logger.info(f"Using per-molecule LogP values from columns {logp1_col} and {logp2_col}")
                            except ValueError as e:
                                logger.error(f"Error converting LogP values: {e}")
                                return False
                        elif logP_filter:  # Use global LogP filter if column-based filters not specified
                            logp_filters = [logP_filter] * len(data_list)
                            logger.info("Using global LogP filter for batch")
                            
                        visualizer.save_image_batch(data_list, output_file_intro=f"polymer_batch_{i+1}", logP_filter=logp_filters)
                else:
                    # plain text file, one SMILES per line
                    with open(input_path, 'r') as f:
                        data_list = [line.strip() for line in f if line.strip()]
                    logp_filters = [logP_filter] * len(data_list) if logP_filter else None
                    visualizer.save_image_batch(data_list, output_file_intro="polymer", logP_filter=logp_filters)
            else:
                # single structure from file
                with open(input_path, 'r') as f:
                    data = f.read().strip()
                visualizer.save_image(data, filename=f"{config.get('output', {}).get('file_prefix','polymer')}.png", logP_filter=logP_filter)
        else:
            # input is a SMILES string
            visualizer.save_image(input_path, filename=f"{config.get('output', {}).get('file_prefix','polymer')}.png", logP_filter=logP_filter)

        logger.info(f"Visualizations saved to: {output_dir}")
        return True
        # except Exception as e:
        #     logger.error(f"Visualization failed: {e}")
        #     return False
    
    def _override_trainer_config(self, config: Dict[str, Any], args):
        """Override trainer configuration with command line arguments."""
        training_config = config.setdefault('training', {})
        
        if args.epochs:
            training_config['epochs'] = args.epochs
        if args.batch_size:
            training_config['batch_size'] = args.batch_size
        if args.learning_rate:
            training_config['learning_rate'] = args.learning_rate
        if args.output_dir:
            training_config['output_dir'] = args.output_dir
        if args.experiment_name:
            training_config['experiment_name'] = args.experiment_name
        if args.debug:
            training_config['debug_batch_prep'] = True
        
        # Device configuration
        if args.device:
            config['device'] = args.device
        
        # Wandb configuration
        if args.wandb:
            logging_config = config.setdefault('logging', {})
            logging_config['use_wandb'] = True
            if args.wandb_project:
                logging_config['project_name'] = args.wandb_project
    
    def _override_sampler_config(self, config: Dict[str, Any], args):
        """Override sampler configuration with command line arguments."""
        generation_config = config.setdefault('generation', {})
        
        if args.num_sequences:
            batch_config = config.setdefault('batch', {})
            batch_config['num_sequences'] = args.num_sequences
        if args.max_length:
            generation_config['max_length'] = args.max_length
        if args.temperature:
            generation_config['temperature'] = args.temperature
        if args.top_p:
            generation_config['top_p'] = args.top_p
        if args.strategy:
            generation_config['decoding_strategy'] = args.strategy
        
        # Experiment configuration
        if args.experiment:
            experiment_config = config.setdefault('experiment', {})
            experiment_config['experiment_dir'] = args.experiment
            if hasattr(args, 'checkpoint_name') and args.checkpoint_name:
                experiment_config['checkpoint_name'] = args.checkpoint_name
        
        # Device configuration
        if args.device:
            config['device'] = args.device

        # Filtering configuration overrides (optional)
        filtering_cfg = config.setdefault('filtering', {})
        disc_cfg = filtering_cfg.setdefault('discriminator', {})

        # Note: argparse replaces '-' with '_' in attribute names
        if getattr(args, 'filtering_discriminator_enabled', None) is not None:
            val = args.filtering_discriminator_enabled.lower()
            disc_cfg['enabled'] = True if val == 'true' else False

        if getattr(args, 'filtering_discriminator_model_type', None):
            disc_cfg['model_type'] = args.filtering_discriminator_model_type

        if getattr(args, 'filtering_discriminator_model_dir', None):
            disc_cfg['model_dir'] = args.filtering_discriminator_model_dir

        if getattr(args, 'filtering_discriminator_logp_threshold', None) is not None:
            disc_cfg['logp_threshold'] = args.filtering_discriminator_logp_threshold

        if getattr(args, 'filtering_do_save_output', None) is not None:
            val = args.filtering_do_save_output.lower()
            filtering_cfg['do_save_output'] = True if val == 'true' else False
    
    def _build_trainer_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build kwargs for trainer creation from config."""
        kwargs = {}
        
        # Training configuration
        training_config = config.get('training', {})
        for key in ['learning_rate', 'batch_size', 'epochs', 'optimizer', 'patience', 
                   'grad_clip', 'output_dir', 'experiment_name', 'debug_batch_prep',
                   'save_best_only', 'save_final', 'save_epoch_checkpoints']:
            if key in training_config:
                kwargs[key] = training_config[key]
        
        # Device configuration
        if 'device' in config:
            kwargs['device'] = config['device']
        
        # Wandb configuration
        logging_config = config.get('logging', {})
        if logging_config.get('use_wandb', False):
            kwargs['use_wandb'] = True
            kwargs['project_name'] = logging_config.get('project_name', 'polymer-training')
        
        return kwargs
    
    def _build_sampler_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build kwargs for sampler creation from config."""
        kwargs = {}
        
        # Generation configuration
        generation_config = config.get('generation', {})
        for key in ['decoding_strategy', 'temperature', 'top_k', 'top_p', 
                   'repetition_penalty', 'max_length', 'min_length', 'early_stopping']:
            if key in generation_config:
                kwargs[key] = generation_config[key]
        
        # Experiment configuration
        experiment_config = config.get('experiment', {})
        if 'experiment_dir' in experiment_config and experiment_config['experiment_dir']:
            kwargs['experiment_dir'] = experiment_config['experiment_dir']
        if 'checkpoint_name' in experiment_config:
            kwargs['checkpoint_name'] = experiment_config['checkpoint_name']
        
        # Device configuration
        if 'device' in config:
            kwargs['device'] = config['device']
        
        return kwargs

    def _build_filter_kwargs(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build kwargs for filtering creation from config."""
        kwargs = {}

        # Filtering configuration
        filtering_config = config.get('filtering', {})
        for key in ['discriminator', 'do_save_output']:
            if key in filtering_config:
                kwargs[key] = filtering_config[key]

        if 'enabled' in filtering_config.get('discriminator', {}):
            if not filtering_config['discriminator']['enabled']:
                return kwargs
            kwargs['discriminator_enabled'] = filtering_config['discriminator']['enabled']
        if 'model_type' in filtering_config.get('discriminator', {}):
            kwargs['discriminator_model_type'] = filtering_config['discriminator']['model_type']
        if 'model_dir' in filtering_config.get('discriminator', {}):
            if not isinstance(filtering_config['discriminator']['model_dir'], str):
                raise ValueError("discriminator.model_dir must be a string path")
            if os.path.exists(filtering_config['discriminator']['model_dir']) is False:
                raise ValueError(f"discriminator.model_dir path does not exist: {filtering_config['discriminator']['model_dir']}")
            kwargs['discriminator_model_dir'] = filtering_config['discriminator']['model_dir']
        if 'logp_threshold' in filtering_config.get('discriminator', {}):
            kwargs['discriminator_logp_threshold'] = filtering_config['discriminator']['logp_threshold']

        kwargs.pop('discriminator', None)
        return kwargs


def create_argument_parser():
    """Create the argument parser for the runner script."""
    parser = argparse.ArgumentParser(
        description="Polymer Training and Sampling Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Training command
    train_parser = subparsers.add_parser('train', help='Train a model')
    train_parser.add_argument('--model', required=True, 
                             choices=['gru', 'lstm', 'transformer', 'vae', 'mamba', 'tcn'],
                             help='Model type to train')
    train_parser.add_argument('--config', help='Custom configuration file')
    train_parser.add_argument('--epochs', type=int, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, help='Batch size')
    train_parser.add_argument('--learning-rate', type=float, help='Learning rate')
    train_parser.add_argument('--output-dir', help='Output directory for experiments')
    train_parser.add_argument('--experiment-name', help='Custom experiment name')
    train_parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    train_parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    train_parser.add_argument('--wandb-project', help='W&B project name')
    train_parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    # Sampling command
    sample_parser = subparsers.add_parser('sample', help='Sample from a trained model')
    group = sample_parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--experiment', help='Path to experiment directory')
    group.add_argument('--checkpoint', help='Path to checkpoint file')
    
    sample_parser.add_argument('--model', help='Model type (required when using --checkpoint)')
    sample_parser.add_argument('--config', help='Custom sampling configuration file')
    sample_parser.add_argument('--checkpoint-name', default='best_model.pth',
                              choices=['best_model.pth', 'final_model.pth', 'latest'],
                              help='Which checkpoint to use from experiment (default: best_model.pth)')
    sample_parser.add_argument('--num-sequences', type=int, help='Number of sequences to generate')
    sample_parser.add_argument('--max-length', type=int, help='Maximum sequence length')
    sample_parser.add_argument('--temperature', type=float, help='Sampling temperature')
    sample_parser.add_argument('--top-p', type=float, help='Top-p (nucleus) sampling parameter')
    sample_parser.add_argument('--strategy', choices=['greedy', 'top_k', 'top_p', 'random'], 
                              help='Decoding strategy')
    sample_parser.add_argument('--device', choices=['auto', 'cuda', 'cpu'], help='Device to use')
    # Additional filtering arguments (override sampler/filtering config)
    sample_parser.add_argument('--filtering-discriminator-enabled',
                               choices=['true', 'false'],
                               help='Enable/disable discriminator filtering (true/false)')
    sample_parser.add_argument('--filtering-discriminator-model-type',
                               help='Discriminator model type (overrides sampler.filtering.discriminator.model_type)')
    sample_parser.add_argument('--filtering-discriminator-model-dir',
                               help='Path to discriminator model directory (overrides sampler.filtering.discriminator.model_dir)')
    sample_parser.add_argument('--filtering-discriminator-logp-threshold', type=float,
                               help='LogP threshold for discriminator filtering (overrides sampler.filtering.discriminator.logp_threshold)')
    sample_parser.add_argument('--filtering-do-save-output',
                               choices=['true', 'false'],
                               help='Whether to save filtered output (overrides sampler.filtering.do_save_output)')
    
    # List experiments command
    list_parser = subparsers.add_parser('list', help='List all experiments')
    
    # Show config command
    config_parser = subparsers.add_parser('show-config', help='Show configuration for a model')
    config_parser.add_argument('--model', required=True,
                              choices=['gru', 'lstm', 'transformer', 'vae', 'mamba', 'tcn'],
                              help='Model type')
    config_parser.add_argument('--mode', required=True, choices=['train', 'sample'],
                              help='Configuration mode')
    
    # Visualization command
    visual_parser = subparsers.add_parser('visual', help='Visualize polymer structures')
    visual_parser.add_argument('--input', required=True,
                              help='SMILES string or path to file containing SMILES strings')
    visual_parser.add_argument('--config', help='Custom visualization configuration file')
    visual_parser.add_argument('--output-dir', help='Output directory for visualizations')
    visual_parser.add_argument('--canvas-width', type=int, help='Width of the canvas for visualization')
    visual_parser.add_argument('--canvas-height', type=int, help='Height of the canvas for visualization')
    visual_parser.add_argument('--figure-width', type=float, help='Width of the figure in inches')
    visual_parser.add_argument('--figure-height', type=float, help='Height of the figure in inches')
    visual_parser.add_argument('--logp-filter', action='store_true', help='Enable logP filtering')
    visual_parser.add_argument('--logp1', type=float, help='LogP value for first fragment')
    visual_parser.add_argument('--logp2', type=float, help='LogP value for second fragment')
    visual_parser.add_argument('--batch', action='store_true', help='Process input as batch (file with multiple SMILES strings)')
    visual_parser.add_argument('--csv-column', help='Column name containing SMILES strings in CSV file (for batch processing)')
    visual_parser.add_argument('--logp1-column', help='Column name containing LogP values for first fragment in CSV file')
    visual_parser.add_argument('--logp2-column', help='Column name containing LogP values for second fragment in CSV file')
    
    return parser


def main():
    """Main entry point for the runner script."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    runner = ModelRunner()
    
    # try:
    if args.command == 'train':
        success = runner.train_model(args)
    elif args.command == 'sample':
        success = runner.sample_from_model(args)
    elif args.command == 'list':
        runner.list_experiments(args)
        success = True
    elif args.command == 'show-config':
        runner.show_config(args)
        success = True
    elif args.command == 'visual':
        success = runner.visualize_structure(args)
    else:
        logger.error(f"Unknown command: {args.command}")
        success = False
    
    if not success:
        sys.exit(1)
            
    # except KeyboardInterrupt:
    #     logger.info("Operation cancelled by user")
    #     sys.exit(1)
    # except Exception as e:
    #     logger.error(f"Unexpected error: {e}")
    #     sys.exit(1)


if __name__ == "__main__":
    main()
