"""
Comprehensive trainer module that adapts to different model architectures and input shapes.
This trainer can handle various input types and model configurations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
import sys
import logging
from datetime import datetime
from typing import Dict, Any, Optional, Union, Tuple
import wandb
from tqdm import tqdm

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach project root
sys.path.insert(0, project_root)

from src.data.dataloader import get_dataloaders
from src.trainer.config import SMILES_DATA_CONFIG


class BaseTrainer:
    """
    Base trainer class that adapts to different model architectures and input configurations.
    """

    def __init__(self,
                 model: nn.Module,
                 model_config: Dict[str, Any],
                 training_config: Dict[str, Any],
                 data_config: Dict[str, Any],
                 device: Optional[str] = None,
                 use_wandb: bool = False,
                 project_name: str = "polymer-training",
                 penalty_config: Dict[str, float] | None = None):
        """
        Initialize the trainer.
        
        Args:
            model: The PyTorch model to train
            model_config: Model-specific configuration
            training_config: Training parameters (lr, epochs, etc.)
            data_config: Data loading configuration
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            use_wandb: Whether to use Weights & Biases for logging
            project_name: W&B project name
        """
        self.model = model
        self.model_config = model_config
        self.training_config = training_config
        self.data_config = data_config
        self.penalty_config = penalty_config if penalty_config is not None else {}

        # Setup output directory structure first
        self._setup_output_directories()

        # Setup logging (after output directories are created)
        self._setup_logging()

        # Device setup
        if device is None:
            self.device = torch.device(
                'cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.model.to(self.device)
        self.logger.info(f"Model moved to device: {self.device}")

        # Logging setup
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=project_name,
                       config={
                           **model_config,
                           **training_config,
                           **data_config
                       })
            self.logger.info(
                f"Weights & Biases initialized for project: {project_name}")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Debug mode
        self.debug_mode = self.training_config.get('debug_batch_prep', False)

        # Setup optimizer and loss function
        self._setup_optimizer()
        self._setup_loss_function()

        # Get data loaders
        self._setup_data_loaders()

        # Detect model type and input requirements
        self._detect_model_characteristics()

        # Debug: Show example batch preparation (optional)
        if self.debug_mode:
            self._debug_batch_preparation()

    def _setup_output_directories(self):
        """
        Setup comprehensive output directory structure for organized experiment management.
        
        Creates the following structure:
        experiments/
        ├── {model_type}_{timestamp}/
        │   ├── checkpoints/
        │   │   ├── best_model.pth
        │   │   ├── final_model.pth
        │   │   └── epoch_checkpoints/
        │   ├── logs/
        │   │   ├── training.log
        │   │   └── metrics.json
        │   ├── config/
        │   │   ├── model_config.json
        │   │   ├── training_config.json
        │   │   └── data_config.json
        │   └── results/
        │       ├── training_history.json
        │       ├── plots/
        │       └── samples/
        """
        # Get base output directory from config
        base_output_dir = self.training_config.get('output_dir', 'experiments')

        # Create experiment-specific directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_type = self.model_config.get('model_type', 'unknown')
        experiment_name = f"{model_type}_{timestamp}"

        # Allow custom experiment name override
        if 'experiment_name' in self.training_config and self.training_config[
                'experiment_name'] is not None:
            experiment_name = self.training_config['experiment_name']

        self.experiment_dir = os.path.join(base_output_dir, experiment_name)

        # Create nested directory structure
        self.dirs = {
            'experiment':
            self.experiment_dir,
            'checkpoints':
            os.path.join(self.experiment_dir, 'checkpoints'),
            'epoch_checkpoints':
            os.path.join(self.experiment_dir, 'checkpoints',
                         'epoch_checkpoints'),
            'logs':
            os.path.join(self.experiment_dir, 'logs'),
            'config':
            os.path.join(self.experiment_dir, 'config'),
            'results':
            os.path.join(self.experiment_dir, 'results'),
            'plots':
            os.path.join(self.experiment_dir, 'results', 'plots'),
            'samples':
            os.path.join(self.experiment_dir, 'results', 'samples')
        }

        # Create all directories
        for dir_name, dir_path in self.dirs.items():
            os.makedirs(dir_path, exist_ok=True)

        # Update training config to use the new checkpoint directory
        self.training_config['save_dir'] = self.dirs['checkpoints']
        self.training_config['logs_dir'] = self.dirs['logs']

        # Save configuration files
        self._save_config_files()

        print(f"📁 Experiment directory created: {self.experiment_dir}")
        print(f"📝 Directory structure:")
        for name, path in self.dirs.items():
            relative_path = os.path.relpath(path, self.experiment_dir)
            print(f"   {name}: {relative_path}")

    def _save_config_files(self):
        """Save all configuration files to the config directory."""
        import json

        configs = {
            'model_config.json': self.model_config,
            'training_config.json': self.training_config,
            'data_config.json': self.data_config
        }

        for filename, config in configs.items():
            config_path = os.path.join(self.dirs['config'], filename)

            # Convert any non-serializable objects to strings
            serializable_config = {}
            for key, value in config.items():
                try:
                    json.dumps(value)  # Test if serializable
                    serializable_config[key] = value
                except (TypeError, ValueError):
                    serializable_config[key] = str(value)

            with open(config_path, 'w') as f:
                json.dump(serializable_config, f, indent=2)

        print(f"💾 Configuration files saved to: {self.dirs['config']}")

    def _setup_logging(self):
        """
        Setup comprehensive logging for the trainer.
        Creates both file and console handlers with appropriate formatting.
        Uses the structured output directory system.
        """
        # Use the logs directory from our structure
        logs_dir = self.dirs['logs']

        # Create log filename
        model_type = self.model_config.get('model_type', 'unknown')
        log_filename = f"training.log"
        self.log_filepath = os.path.join(logs_dir, log_filename)

        # Create logger
        self.logger = logging.getLogger(f"trainer_{model_type}")
        self.logger.setLevel(logging.INFO)

        # Clear any existing handlers to avoid duplicates
        self.logger.handlers.clear()

        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        simple_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

        # File handler (detailed logging)
        file_handler = logging.FileHandler(self.log_filepath, mode='w')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)

        # Console handler (simplified logging)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)

        # Log the initialization
        self.logger.info("=" * 80)
        self.logger.info(
            f"TRAINING SESSION STARTED - {model_type.upper()} MODEL")
        self.logger.info("=" * 80)
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info(f"Log file: {self.log_filepath}")
        self.logger.info(f"Checkpoints directory: {self.dirs['checkpoints']}")
        self.logger.info(f"Model type: {model_type}")
        self.logger.info(f"Training config: {self.training_config}")
        self.logger.info(f"Model config: {self.model_config}")

        # Create metrics log file path for structured logging
        self.metrics_log_path = os.path.join(logs_dir, 'metrics.json')

    def _debug_batch_preparation(self):
        """Debug function to show how batch preparation works."""
        self.logger.info("🔍 DEBUG: Batch Preparation Example")
        self.logger.info("=" * 50)

        # Get a sample batch
        sample_batch = next(iter(self.train_loader))
        self.logger.info(f"Original batch shape: {sample_batch.shape}")
        self.logger.info(
            f"Sample sequence (first 10 tokens): {sample_batch[0, :10].tolist()}"
        )

        # Prepare the batch
        inputs, targets = self._prepare_batch(sample_batch)
        self.logger.info(f"\nAfter preparation:")
        self.logger.info(f"Input shape: {inputs.shape}")
        self.logger.info(f"Target shape: {targets.shape}")
        self.logger.info(
            f"Input sequence (first 10): {inputs[0, :10].tolist()}")
        self.logger.info(
            f"Target sequence (first 10): {targets[0, :10].tolist()}")

        if self.is_seq2seq:
            self.logger.info(f"\n📚 Training Logic:")
            self.logger.info(
                f"At each position i, model sees input[:i+1] and predicts target[i]"
            )
            self.logger.info(
                f"Position 0: input=[{inputs[0, 0].item()}] → predict target={targets[0, 0].item()}"
            )
            self.logger.info(
                f"Position 1: input=[{inputs[0, 0].item()}, {inputs[0, 1].item()}] → predict target={targets[0, 1].item()}"
            )
            self.logger.info(
                f"Position 2: input=[{inputs[0, 0].item()}, {inputs[0, 1].item()}, {inputs[0, 2].item()}] → predict target={targets[0, 2].item()}"
            )

        # Test model forward pass with debug info
        self.logger.info(f"\n🚀 Model Forward Pass Debug:")
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(inputs)
            main_output, aux_outputs = self._standardize_model_output(outputs)
            self.logger.info(f"Raw outputs type: {type(outputs)}")
            if isinstance(outputs, tuple):
                self.logger.info(f"Tuple length: {len(outputs)}")
                for i, item in enumerate(outputs):
                    if item is not None:
                        self.logger.info(
                            f"  Item {i}: {type(item)} shape {item.shape if hasattr(item, 'shape') else 'N/A'}"
                        )
                    else:
                        self.logger.info(f"  Item {i}: None")
            else:
                self.logger.info(f"Single tensor shape: {outputs.shape}")

            self.logger.info(
                f"Standardized main output shape: {main_output.shape}")
            self.logger.info(f"Auxiliary outputs: {aux_outputs}")

            # Test loss computation
            loss = self._compute_loss(outputs, targets, inputs)
            self.logger.info(f"Loss computed successfully: {loss.item():.4f}")

        self.model.train()
        self.logger.info("=" * 50)

    def _setup_optimizer(self):
        """Setup optimizer based on training config."""
        optimizer_type = self.training_config.get('optimizer', 'adam')
        lr = self.training_config.get('learning_rate', 1e-3)
        weight_decay = self.training_config.get('weight_decay', 0.0)

        # Convert string values to appropriate numeric types
        if isinstance(lr, str):
            lr = float(lr)
        if isinstance(weight_decay, str):
            weight_decay = float(weight_decay)

        if optimizer_type.lower() == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            self.optimizer = optim.AdamW(self.model.parameters(),
                                         lr=lr,
                                         weight_decay=weight_decay)
        elif optimizer_type.lower() == 'sgd':
            momentum = self.training_config.get('momentum', 0.9)
            # Convert string values to appropriate numeric types
            if isinstance(momentum, str):
                momentum = float(momentum)
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=lr,
                                       momentum=momentum,
                                       weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

        self.logger.info(
            f"Optimizer configured: {optimizer_type} with lr={lr}, weight_decay={weight_decay}"
        )

    def _setup_loss_function(self):
        """Setup loss function based on model type."""
        loss_type = self.training_config.get('loss_function', 'cross_entropy')

        if loss_type == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=0)  # Ignore padding tokens
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_type == 'bce':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

        self.logger.info(f"Loss function configured: {loss_type}")

    def _setup_data_loaders(self):
        """Setup data loaders."""
        batch_size = self.training_config.get('batch_size', 32)

        # Convert string values to appropriate numeric types
        if isinstance(batch_size, str):
            batch_size = int(batch_size)

        self.train_loader, self.val_loader, self.test_loader = get_dataloaders(
            DATA_CONFIG=self.data_config,
            batch_size=batch_size,
            PENALTY_CONFIG=self.penalty_config)

        # Safely get dataset sizes
        def get_dataset_size(loader):
            try:
                if hasattr(loader.dataset, '__len__'):
                    return len(loader.dataset)
                else:
                    return "Unknown"
            except:
                return "Unknown"

        train_size = get_dataset_size(self.train_loader)
        val_size = get_dataset_size(self.val_loader)
        test_size = get_dataset_size(self.test_loader)
        self.logger.info(
            f"Dataset sizes - Train: {train_size}, Val: {val_size}, Test: {test_size}"
        )
        self.logger.info(f"Batch size: {batch_size}")

    def _detect_model_characteristics(self):
        """
        Detect model characteristics to determine training approach.
        """
        model_name = self.model.__class__.__name__.lower()

        # Check the actual model output to determine if it's generative
        self.is_generative = self._check_if_generative()

        # All generative models use sequence-to-sequence (autoregressive) training
        # This includes GRU, LSTM, Transformer (now generative), VAE, Mamba, TCN
        self.is_seq2seq = self.is_generative

        # Model type flags
        self.is_transformer = 'transformer' in model_name
        self.is_vae = 'vae' in model_name
        self.is_mamba = 'mamba' in model_name
        self.is_tcn = 'tcn' in model_name

        self.logger.info(f"Model type detected: {model_name}")
        self.logger.info(f"Generative model: {self.is_generative}")
        self.logger.info(
            f"Sequence-to-sequence (autoregressive): {self.is_seq2seq}")
        self.logger.info(f"Transformer-based: {self.is_transformer}")
        self.logger.info(f"VAE-based: {self.is_vae}")
        self.logger.info(f"Mamba-based: {self.is_mamba}")
        self.logger.info(f"TCN-based: {self.is_tcn}")

    def _check_if_generative(self) -> bool:
        """
        Check if the model is generative by examining its output shape.
        
        Returns:
            True if model outputs vocab_size logits (generative)
            False if model outputs scalar/small tensor (discriminative)
        """
        try:
            # Create a dummy input to check output shape
            dummy_input = torch.zeros(1, 10, dtype=torch.long).to(self.device)
            self.model.eval()

            with torch.no_grad():
                output = self.model(dummy_input)

                if isinstance(output, tuple):
                    # VAE case: check reconstruction output
                    output = output[0]

                self.logger.info(f"Model output shape: {output.shape}")

                # Check if output has vocab_size dimension (indicating generative model)
                if len(output.shape) == 3:
                    # Shape: (batch_size, seq_len, vocab_size) - generative sequence model
                    vocab_size = output.size(-1)
                    self.logger.info(
                        f"3D output detected with vocab_size: {vocab_size}")
                    return vocab_size > 100  # Reasonable threshold for vocab size
                elif len(output.shape) == 2:
                    # Could be (batch_size, vocab_size) for single prediction
                    # or (batch_size, small_dim) for classification
                    vocab_size = output.size(-1)
                    self.logger.info(
                        f"2D output detected with last dim: {vocab_size}")
                    return vocab_size > 100  # Reasonable threshold for vocab size
                else:
                    # 1D or other shapes suggest non-generative
                    self.logger.info(f"Non-generative output shape detected")
                    return False

        except Exception as e:
            self.logger.warning(
                f"Could not determine if model is generative: {e}")
            # Default fallback based on model name - all models we have are generative
            model_name = self.model.__class__.__name__.lower()
            is_generative = any(
                name in model_name for name in
                ['gru', 'lstm', 'transformer', 'vae', 'mamba', 'tcn'])
            self.logger.info(
                f"Fallback: Assuming {model_name} is generative: {is_generative}"
            )
            return is_generative

        finally:
            self.model.train()

    def _prepare_batch(
            self, batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare batch data based on model requirements.
        
        Args:
            batch: Input batch tensor of shape (batch_size, seq_len)
            
        Returns:
            Tuple of (input_data, target_data)
            
        For generative models (autoregressive):
            - input_data: sequence[:-1] - what the model sees as context
            - target_data: sequence[1:] - what the model should predict (next tokens)
            
        For discriminative models (classification/regression):
            - input_data: full sequence
            - target_data: same sequence or separate labels
            
        Example for generative:
            Original: [<bos>, C, C, O, H, <eos>]
            Input:    [<bos>, C, C, O, H]        (model input)
            Target:   [C, C, O, H, <eos>]        (expected predictions)
        """
        batch = batch.to(self.device)

        if self.debug_mode:
            self.logger.debug(f"\nBatch preparation:")
            self.logger.debug(f"Input batch shape: {batch.shape}")
            self.logger.debug(f"Model is generative: {self.is_generative}")
            self.logger.debug(f"Model is seq2seq: {self.is_seq2seq}")

        # For generative models that use autoregressive training
        if self.is_generative and self.is_seq2seq:
            # For autoregressive text generation: predict next token at each position
            # This implements "teacher forcing" training strategy
            input_data = batch[:, :
                               -1]  # Remove last token (no next token to predict)
            target_data = batch[:,
                                1:]  # Remove first token (shift targets left by 1)

            # Ensure we have valid sequences
            if input_data.size(1) == 0:
                raise ValueError(
                    "Input sequences too short for autoregressive training")

            if self.debug_mode:
                self.logger.debug(f"Autoregressive mode:")
                self.logger.debug(
                    f"  Input shape: {input_data.shape} (tokens[:-1])")
                self.logger.debug(
                    f"  Target shape: {target_data.shape} (tokens[1:])")
                self.logger.debug(
                    f"  Example input: {input_data[0][:10].tolist()}")
                self.logger.debug(
                    f"  Example target: {target_data[0][:10].tolist()}")

        # For discriminative models (classification/regression)
        elif not self.is_generative:
            input_data = batch
            # For property prediction, targets would typically come from labels
            # For now, using same sequence (could be used for reconstruction tasks)
            target_data = batch

            if self.debug_mode:
                self.logger.debug(f"Discriminative mode:")
                self.logger.debug(f"  Input shape: {input_data.shape}")
                self.logger.debug(f"  Target shape: {target_data.shape}")

        # For other generative models (like VAE) that reconstruct the input
        else:
            input_data = batch
            target_data = batch  # Reconstruction target

            if self.debug_mode:
                self.logger.debug(f"Reconstruction mode:")
                self.logger.debug(f"  Input shape: {input_data.shape}")
                self.logger.debug(f"  Target shape: {target_data.shape}")

        return input_data, target_data

    def _standardize_model_output(
        self, outputs: Union[torch.Tensor, Tuple]
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Standardize model outputs to a consistent format.
        
        Args:
            outputs: Raw model outputs (could be tensor or tuple)
            
        Returns:
            Tuple of (main_output, auxiliary_outputs)
            - main_output: Primary tensor for loss computation (logits/predictions)
            - auxiliary_outputs: Dictionary of auxiliary outputs (e.g., VAE parameters)
        """
        if isinstance(outputs, torch.Tensor):
            # Single tensor output (e.g., Transformer)
            return outputs, None

        elif isinstance(outputs, tuple):
            if len(outputs) == 2:
                # Two outputs: could be (logits, hidden) or (logits, None)
                main_output, aux = outputs
                if aux is None:
                    # (logits, None) - e.g., Mamba, TCN
                    return main_output, None
                else:
                    # (logits, hidden_state) - e.g., GRU, LSTM
                    return main_output, {'hidden_state': aux}

            elif len(outputs) == 3:
                # Three outputs: VAE case (reconstruction, mu, log_var)
                reconstruction, mu, log_var = outputs
                return reconstruction, {'mu': mu, 'log_var': log_var}

            else:
                raise ValueError(f"Unexpected tuple length: {len(outputs)}")

        else:
            raise ValueError(f"Unexpected output type: {type(outputs)}")

    def _compute_loss(
            self,
            outputs: Union[torch.Tensor, Tuple],
            targets: torch.Tensor,
            inputs: torch.Tensor,
            penalties: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Compute loss based on model type and outputs.
        
        Args:
            outputs: Model outputs (could be tensor or tuple)
            targets: Target data
            inputs: Input data (needed for some loss computations)
            penalties: Dictionary of penalty terms to be added to the loss
        
        Returns:
            Total loss
        """
        # Standardize outputs to consistent format
        main_output, aux_outputs = self._standardize_model_output(outputs)

        # Initialize custom loss as a zero tensor matching the penalty vector shape
        custom_loss = torch.zeros_like(next(iter(penalties.values()))) if penalties else torch.tensor(0.0, device=main_output.device)
        custom_loss = custom_loss.to(main_output.device)

        if penalties is not None:
            for key, penalty in penalties.items():
                assert key in self.penalty_config, f"Unexpected penalty key: {key}"
                weight = self.penalty_config[key]
                penalty = penalty.to(main_output.device)
                custom_loss += weight * penalty

        # Handle VAE case with auxiliary loss terms
        if self.is_vae and aux_outputs is not None and 'mu' in aux_outputs and 'log_var' in aux_outputs:
            mu = aux_outputs['mu']
            log_var = aux_outputs['log_var']

            # Reconstruction loss
            if self.is_seq2seq:
                # Sequence-to-sequence: flatten for cross-entropy
                recon_loss = self.criterion(
                    main_output.reshape(-1, main_output.size(-1)),
                    targets.reshape(-1))
            else:
                recon_loss = self.criterion(main_output, targets)

            # KL divergence loss
            kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            kl_weight = self.training_config.get('kl_weight', 0.1)

            total_loss = recon_loss + kl_weight * kl_loss + custom_loss.sum()

            # Log VAE-specific losses
            if self.global_step % 100 == 0:  # Log every 100 steps to avoid spam
                self.logger.debug(
                    f"VAE losses - Recon: {recon_loss.item():.4f}, KL: {kl_loss.item():.4f}, KL_weight: {kl_weight}, Custom: {custom_loss.sum().item():.4f}"
                )

            if self.use_wandb:
                wandb.log({
                    'train/recon_loss': recon_loss.item(),
                    'train/kl_loss': kl_loss.item(),
                    'train/kl_weight': kl_weight,
                    'train/custom_loss': custom_loss.sum().item()
                })

            return total_loss

        else:
            # Standard generative models (GRU, LSTM, Transformer, Mamba, TCN)
            if not isinstance(main_output, torch.Tensor):
                raise ValueError(
                    f"Expected tensor output, got {type(main_output)}")

            if self.is_seq2seq and len(main_output.shape) == 3:
                # Sequence-to-sequence: flatten for cross-entropy
                # main_output shape: (batch_size, seq_len, vocab_size)
                # targets shape: (batch_size, seq_len)
                loss = self.criterion(
                    main_output.reshape(-1, main_output.size(-1)),
                    targets.reshape(-1))
            elif len(main_output.shape) == 2:
                # 2D output: direct loss computation
                loss = self.criterion(main_output, targets)
            else:
                raise ValueError(
                    f"Unexpected output shape: {main_output.shape}")

            total_loss = loss + custom_loss.sum()

            # Log standard losses
            if self.global_step % 50== 0:
                self.logger.debug(
                    f"Standard losses - Main: {loss.item():.4f}, Custom: {custom_loss.sum().item():.4f}"
                )

            if self.use_wandb:
                wandb.log({
                    'train/main_loss': loss.item(),
                    'train/custom_loss': custom_loss.sum().item()
                })

            return total_loss

    def train_epoch(self) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        progress_bar = tqdm(self.train_loader,
                            desc=f"Epoch {self.current_epoch + 1}")

        for batch in progress_bar:
            self.optimizer.zero_grad()

            sequences, penalties = batch

            # Prepare batch data
            inputs, targets = self._prepare_batch(sequences)

            # Forward pass
            outputs = self.model(inputs)

            # Compute loss
            loss = self._compute_loss(outputs,
                                      targets,
                                      inputs,
                                      penalties=penalties)

            # Backward pass
            loss.backward()

            # Gradient clipping if specified
            if 'grad_clip' in self.training_config:
                grad_clip_value = self.training_config['grad_clip']
                # Convert string values to appropriate numeric types
                if isinstance(grad_clip_value, str):
                    grad_clip_value = float(grad_clip_value)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               grad_clip_value)

            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})

            # Log to wandb if enabled
            if self.use_wandb and self.global_step % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/step': self.global_step
                })

        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}

    def validate(self) -> Dict[str, float]:
        """
        Validate the model.
        
        Returns:
            Dictionary with validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                sequences, penalties = batch
                inputs, targets = self._prepare_batch(sequences)
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets, inputs, penalties=penalties)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'val_loss': avg_loss}

    def test(self) -> Dict[str, float]:
        """
        Test the model.
        
        Returns:
            Dictionary with test metrics
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.test_loader:
                sequence, penalties = batch
                inputs, targets = self._prepare_batch(sequence)
                outputs = self.model(inputs)
                loss = self._compute_loss(outputs, targets, inputs, penalties=penalties)

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        return {'test_loss': avg_loss}

    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """Save model checkpoint with enhanced metadata."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'experiment_dir': self.experiment_dir,
            'timestamp': datetime.now().isoformat(),
            'global_step': self.global_step
        }

        torch.save(checkpoint, filepath)
        checkpoint_type = "best" if is_best else "regular"
        self.logger.info(
            f"Saved {checkpoint_type} checkpoint: {os.path.relpath(filepath, self.experiment_dir)}"
        )

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.global_step = checkpoint.get('global_step', 0)

        self.logger.info(
            f"Checkpoint loaded from epoch {self.current_epoch}, filepath: {os.path.relpath(filepath, self.experiment_dir)}"
        )

    def _save_training_metrics(self, metrics: Dict[str, Any]):
        """Save training metrics to JSON file for later analysis."""
        import json

        # Load existing metrics if file exists
        if os.path.exists(self.metrics_log_path):
            with open(self.metrics_log_path, 'r') as f:
                all_metrics = json.load(f)
        else:
            all_metrics = {
                'experiment_info': {
                    'model_type': self.model_config.get('model_type'),
                    'start_time': datetime.now().isoformat(),
                    'experiment_dir': self.experiment_dir
                },
                'epochs': []
            }

        # Add current epoch metrics
        epoch_metrics = {
            'epoch': self.current_epoch + 1,
            'global_step': self.global_step,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        all_metrics['epochs'].append(epoch_metrics)

        # Save updated metrics
        with open(self.metrics_log_path, 'w') as f:
            json.dump(all_metrics, f, indent=2)

    def _save_training_history(self, history: Dict[str, Any]):
        """Save final training history to results directory."""
        import json

        history_path = os.path.join(self.dirs['results'],
                                    'training_history.json')

        # Add experiment metadata
        enhanced_history = {
            'experiment_info': {
                'model_type': self.model_config.get('model_type'),
                'experiment_dir': self.experiment_dir,
                'start_time': datetime.now().isoformat(),
                'model_params':
                sum(p.numel() for p in self.model.parameters()),
            },
            'config': {
                'model_config': self.model_config,
                'training_config': self.training_config,
                'data_config': self.data_config
            },
            'results': history
        }

        with open(history_path, 'w') as f:
            json.dump(enhanced_history, f, indent=2)

        self.logger.info(
            f"Training history saved: {os.path.relpath(history_path, self.experiment_dir)}"
        )

    def train(self, epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Main training loop.
        
        Args:
            epochs: Number of epochs to train (overrides config if provided)
            
        Returns:
            Training history dictionary
        """
        if epochs is None:
            epochs = self.training_config.get('epochs', 10)

        # Ensure epochs is a valid integer
        if isinstance(epochs, str):
            epochs = int(epochs)
        elif not isinstance(epochs, int):
            epochs = 10  # Default fallback

        patience = self.training_config.get('patience', 10)
        # Convert string values to appropriate numeric types
        if isinstance(patience, str):
            patience = int(patience)
        save_dir = self.dirs['checkpoints']  # Use structured directory

        # Training session start
        history: Dict[str, Any] = {'train_loss': [], 'val_loss': []}
        model_params = sum(p.numel() for p in self.model.parameters())

        self.logger.info("=" * 60)
        self.logger.info("STARTING TRAINING SESSION")
        self.logger.info("=" * 60)
        self.logger.info(f"Training for {epochs} epochs")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {model_params:,}")
        self.logger.info(f"Patience: {patience}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info(f"Checkpoints directory: {save_dir}")
        self.logger.info("=" * 60)

        # Start training timer
        training_start_time = time.time()

        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()

            self.logger.info(f"Starting Epoch {epoch + 1}/{epochs}")

            # Training
            train_metrics = self.train_epoch()

            # Validation
            val_metrics = self.validate()

            # Update history
            history['train_loss'].append(train_metrics['train_loss'])
            history['val_loss'].append(val_metrics['val_loss'])

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time

            # Combine metrics for saving
            epoch_metrics = {
                **train_metrics,
                **val_metrics, 'epoch_time': epoch_time
            }

            # Save metrics to JSON
            self._save_training_metrics(epoch_metrics)

            # Log progress
            self.logger.info(
                f"Epoch {epoch + 1}/{epochs} completed in {epoch_time:.2f}s - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch + 1,
                    'train/epoch_loss': train_metrics['train_loss'],
                    'val/epoch_loss': val_metrics['val_loss'],
                    'epoch_time': epoch_time
                })

            # Early stopping and checkpointing
            is_best = val_metrics['val_loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['val_loss']
                self.patience_counter = 0

                # Save best checkpoint if enabled
                if self.training_config.get('save_best_only', True):
                    best_checkpoint_path = os.path.join(
                        save_dir, 'best_model.pth')
                    self.save_checkpoint(best_checkpoint_path, is_best=True)
                    self.logger.info(
                        f"🏆 New best model! Val Loss: {val_metrics['val_loss']:.4f}"
                    )
            else:
                self.patience_counter += 1
                self.logger.info(
                    f"No improvement. Patience: {self.patience_counter}/{patience}"
                )

            # Optional: Save checkpoint every epoch (if save_best_only is False)
            if not self.training_config.get('save_best_only', True):
                epoch_checkpoint_path = os.path.join(
                    self.dirs['epoch_checkpoints'],
                    f'checkpoint_epoch_{epoch + 1}.pth')
                self.save_checkpoint(epoch_checkpoint_path, is_best=is_best)

            # Early stopping
            if self.patience_counter >= patience:
                self.logger.warning(
                    f"🛑 Early stopping triggered after {epoch + 1} epochs")
                break

        # Calculate total training time
        total_training_time = time.time() - training_start_time

        # Final test evaluation
        self.logger.info("Running final test evaluation...")
        test_metrics = self.test()

        # Log final results
        self.logger.info("=" * 60)
        self.logger.info("TRAINING COMPLETED")
        self.logger.info("=" * 60)
        self.logger.info(
            f"Total training time: {total_training_time:.2f}s ({total_training_time/60:.2f} minutes)"
        )
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info(f"Final test loss: {test_metrics['test_loss']:.4f}")
        self.logger.info(f"Experiment directory: {self.experiment_dir}")
        self.logger.info(
            f"Log file: {os.path.relpath(self.log_filepath, self.experiment_dir)}"
        )

        # Save final checkpoint if enabled
        if self.training_config.get('save_final', True):
            final_checkpoint_path = os.path.join(save_dir, 'final_model.pth')
            self.save_checkpoint(final_checkpoint_path, is_best=False)

        if self.use_wandb:
            wandb.log({
                'test/final_loss': test_metrics['test_loss'],
                'training/total_time': total_training_time,
                'training/best_val_loss': self.best_val_loss
            })
            wandb.finish()

        # Prepare final history
        history['test_loss'] = test_metrics['test_loss']
        history['total_training_time'] = total_training_time
        history['best_val_loss'] = self.best_val_loss
        history['experiment_dir'] = self.experiment_dir

        # Save comprehensive training history
        self._save_training_history(history)

        self.logger.info("=" * 60)
        return history


def create_trainer(model_name: str, vocab_size: int, **kwargs) -> BaseTrainer:
    """
    Factory function to create a trainer for a specific model.
    
    Args:
        model_name: Name of the model ('gru', 'lstm', 'transformer', etc.)
        vocab_size: Vocabulary size for the model
        **kwargs: Additional arguments for trainer configuration
        
    Returns:
        Configured trainer instance
    """
    # Import models dynamically to avoid circular imports
    if model_name.lower() == 'gru':
        from src.models.gru.model import GRUModel
        from src.models.gru.config import embedding_dim, hidden_dim, num_layers, dropout

        model = GRUModel(vocab_size=vocab_size)
        model_config = {
            'model_type': 'gru',
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }

    elif model_name.lower() == 'lstm':
        from src.models.lstm.model import LSTMModel
        from src.models.lstm.config import MODEL_CONFIG

        model = LSTMModel(vocab_size=vocab_size)
        model_config = {
            'model_type': 'lstm',
            'vocab_size': vocab_size,
            **MODEL_CONFIG
        }

    elif model_name.lower() == 'transformer':
        from src.models.transformer.model import TransformerModel
        from src.models.transformer.config import MODEL_CONFIG

        model = TransformerModel(vocab_size=vocab_size)
        model_config = {
            'model_type': 'transformer',
            'vocab_size': vocab_size,
            **MODEL_CONFIG
        }

    elif model_name.lower() == 'vae':
        from src.models.vae.model import VAE
        from src.models.vae.config import embedding_dim, hidden_dim, z_dim, num_layers, dropout

        model = VAE(vocab_size=vocab_size)
        model_config = {
            'model_type': 'vae',
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_dim': hidden_dim,
            'z_dim': z_dim,
            'num_layers': num_layers,
            'dropout': dropout
        }

    elif model_name.lower() == 'mamba':
        from src.models.mamba.model import MambaModel
        from src.models.mamba.config import d_model, n_layer, d_state, d_conv, expand

        model = MambaModel(vocab_size=vocab_size)
        model_config = {
            'model_type': 'mamba',
            'vocab_size': vocab_size,
            'd_model': d_model,
            'n_layer': n_layer,
            'd_state': d_state,
            'd_conv': d_conv,
            'expand': expand
        }

    elif model_name.lower() == 'tcn':
        from src.models.tcn.model import TCNModel
        from src.models.tcn.config import embedding_dim, num_channels, kernel_size, dropout

        model = TCNModel(vocab_size=vocab_size, embedding_dim=embedding_dim, num_channels=num_channels)
        model_config = {
            'model_type': 'tcn',
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'num_channels': num_channels,
            'kernel_size': kernel_size,
            'dropout': dropout
        }

    else:
        # raise ValueError(f"Unsupported model: {model_name}")
        available_models = ['gru', 'lstm', 'transformer', 'vae', 'mamba', 'tcn']
        raise ValueError(f"Unsupported model: {model_name}. Available models: {available_models}")

    # Default training configuration
    training_config = {
        'learning_rate': kwargs.get('learning_rate', 1e-3),
        'batch_size': kwargs.get('batch_size', 32),
        'epochs': kwargs.get('epochs', 10),
        'optimizer': kwargs.get('optimizer', 'adam'),
        'patience': kwargs.get('patience', 5),
        'grad_clip': kwargs.get('grad_clip', 1.0),

        # Output directory configuration
        'output_dir': kwargs.get('output_dir', 'experiments'),  # Base output directory
        'experiment_name': kwargs.get('experiment_name', None),  # Custom experiment name (optional)

        # Legacy support (will be overridden by structured directories)
        'save_dir': kwargs.get('save_dir', f'checkpoints/{model_name}'),
        'logs_dir': kwargs.get('logs_dir', 'logs'),

        # Checkpoint configuration
        'save_best_only': kwargs.get('save_best_only', True),  # Only save best checkpoint
        'save_final': kwargs.get('save_final', True),  # Save final checkpoint
        'save_epoch_checkpoints': kwargs.get('save_epoch_checkpoints', False),  # Save every epoch

        # Debug configuration
        'debug_batch_prep': kwargs.get('debug_batch_prep', False),  # Enable batch preparation debugging

        **kwargs.get('training_config', {})
    }

    # Data configuration
    data_config = kwargs.get('data_config', SMILES_DATA_CONFIG)

    # Penalty configuration (if any)
    penalty_config = kwargs.get('penalty_config', None)

    project_name = kwargs.get('project_name', 'not found')
    print('project name >>', project_name)

    return BaseTrainer(
        model=model,
        model_config=model_config,
        training_config=training_config,
        data_config=data_config,
        penalty_config=penalty_config,
        device=kwargs.get('device'),
        use_wandb=kwargs.get('use_wandb', False),
        project_name=kwargs.get('project_name', 'polymer-training')
    )


if __name__ == "__main__":
    # Example usage with enhanced directory structure and logging
    trainer = create_trainer(
        model_name="tcn",
        vocab_size=28996,
        debug_batch_prep=True,  # Enable debug logging
        output_dir="experiments",  # Base output directory
        experiment_name="tcn_polymer_experiment_v1",  # Custom experiment name
        save_epoch_checkpoints=False,  # Don't save every epoch to save space
        epochs=1
    )

    print("\n" + "="*60)
    print("🚀 TRAINER INITIALIZED WITH STRUCTURED OUTPUT DIRECTORIES")
    print("="*60)
    print(f"📁 Experiment directory: {trainer.experiment_dir}")
    print(f"📝 Log file: {os.path.relpath(trainer.log_filepath, trainer.experiment_dir)}")
    print(f"💾 Checkpoints: {os.path.relpath(trainer.dirs['checkpoints'], trainer.experiment_dir)}")
    print(f"📊 Results: {os.path.relpath(trainer.dirs['results'], trainer.experiment_dir)}")
    print(f"⚙️  Config: {os.path.relpath(trainer.dirs['config'], trainer.experiment_dir)}")
    print("="*60)

    # Start training (logs and checkpoints will be organized automatically)
    history = trainer.train()

    print("\n" + "="*60)
    print("✅ TRAINING COMPLETED!")
    print("="*60)
    print(f"📁 All outputs saved to: {trainer.experiment_dir}")
    print(f"📝 Training log: {trainer.log_filepath}")
    print(f"📊 Training history: {os.path.join(trainer.dirs['results'], 'training_history.json')}")
    print(f"📈 Metrics log: {trainer.metrics_log_path}")
    print("="*60)
