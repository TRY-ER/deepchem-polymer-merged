"""
Comprehensive sampler module for generating molecular sequences using trained models.
This sampler can load checkpoints and perform inference with various text generation strategies.
"""

import glob
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(
    os.path.dirname(current_dir)
)  # Go up two levels to reach project root
sys.path.insert(0, project_root)

from src.sampler.config import (
    DEFAULT_SAMPLING_CONFIG,
    MODEL_SAMPLING_CONFIGS,
    STARTER_TEXTS,
)
from src.tokenizer import Tokenizer


class Sampler:
    """
    A comprehensive sampler for generating sequences using trained models.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        sampling_config: Optional[Dict[str, Any]] = None,
        filtering_config: Optional[Dict[str, Any]] = None,
        auto_detect_best: bool = True,
    ):
        """
        Initialize the sampler.

        Args:
            checkpoint_path: Path to model checkpoint or experiment directory
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
            sampling_config: Configuration for sampling parameters
            auto_detect_best: If True and checkpoint_path is a directory,
                             automatically find and use the best checkpoint
        """
        self.auto_detect_best = auto_detect_best

        # Resolve checkpoint path (could be file or experiment directory)
        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)

        # Device setup
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load checkpoint and model
        self.checkpoint = self._load_checkpoint()
        self.model = self._load_model()
        self.model_config = self.checkpoint["model_config"]
        self.model_type = self.model_config["model_type"]

        # Get experiment directory if available
        self.experiment_dir = self.checkpoint.get("experiment_dir", None)

        # Setup tokenizer
        self.tokenizer = Tokenizer()
        self.vocab_size = self.tokenizer.tokenizer.vocab_size

        # Setup sampling configuration
        self.sampling_config = DEFAULT_SAMPLING_CONFIG.copy()
        if sampling_config:
            self.sampling_config.update(sampling_config)

        # Apply model-specific adjustments
        if self.model_type in MODEL_SAMPLING_CONFIGS:
            self.sampling_config.update(MODEL_SAMPLING_CONFIGS[self.model_type])

        self.filtering_config = filtering_config if filtering_config else {}

        print(f"Sampler initialized for {self.model_type} model")
        print(f"Checkpoint: {os.path.basename(self.checkpoint_path)}")
        if self.experiment_dir:
            print(f"Experiment directory: {self.experiment_dir}")
        print(f"Device: {self.device}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def _resolve_checkpoint_path(self, path: str) -> str:
        """
        Resolve checkpoint path - handle both direct file paths and experiment directories.

        Args:
            path: Either a direct checkpoint file path or an experiment directory

        Returns:
            Resolved checkpoint file path
        """
        if os.path.isfile(path):
            # Direct checkpoint file
            print(f"Loading checkpoint file: {path}")
            return path
        elif os.path.isdir(path):
            # Experiment directory - find checkpoint
            print(f"Searching for checkpoints in experiment directory: {path}")
            return self._find_best_checkpoint_in_directory(path)
        else:
            # Try to find experiment directory or checkpoint file
            potential_paths = [
                path,  # Original path
                os.path.join("experiments", path),  # Assume it's an experiment name
                os.path.join(
                    "experiments", path, "checkpoints"
                ),  # Direct to checkpoints dir
                path + ".pth",  # Add .pth extension
            ]

            for potential_path in potential_paths:
                if os.path.isfile(potential_path):
                    print(f"Found checkpoint file: {potential_path}")
                    return potential_path
                elif os.path.isdir(potential_path):
                    print(f"Found experiment directory: {potential_path}")
                    return self._find_best_checkpoint_in_directory(potential_path)

            raise FileNotFoundError(f"Cannot find checkpoint at: {path}")

    def _find_best_checkpoint_in_directory(self, directory: str) -> str:
        """
        Find the best checkpoint in an experiment directory.

        Args:
            directory: Experiment directory path

        Returns:
            Path to the best checkpoint file
        """
        # Look for checkpoints directory within experiment directory
        checkpoints_dir = os.path.join(directory, "checkpoints")
        if not os.path.isdir(checkpoints_dir):
            # Maybe the directory itself is the checkpoints directory
            checkpoints_dir = directory

        if self.auto_detect_best:
            # Priority order for checkpoint selection
            checkpoint_files = ["best_model.pth", "final_model.pth"]

            for checkpoint_file in checkpoint_files:
                checkpoint_path = os.path.join(checkpoints_dir, checkpoint_file)
                if os.path.isfile(checkpoint_path):
                    print(f"Found {checkpoint_file} checkpoint")
                    return checkpoint_path

            # Fallback: search for any .pth files
            pth_files = glob.glob(os.path.join(checkpoints_dir, "*.pth"))
            if pth_files:
                # Use the newest file by modification time
                newest_checkpoint = max(pth_files, key=os.path.getmtime)
                print(f"Using newest checkpoint: {os.path.basename(newest_checkpoint)}")
                return newest_checkpoint

        # Search in epoch_checkpoints subdirectory
        epoch_checkpoints_dir = os.path.join(checkpoints_dir, "epoch_checkpoints")
        if os.path.isdir(epoch_checkpoints_dir):
            pth_files = glob.glob(os.path.join(epoch_checkpoints_dir, "*.pth"))
            if pth_files:
                # Use the newest file by modification time
                newest_checkpoint = max(pth_files, key=os.path.getmtime)
                print(
                    f"Using newest epoch checkpoint: {os.path.basename(newest_checkpoint)}"
                )
                return newest_checkpoint

        raise FileNotFoundError(f"No checkpoint files found in: {directory}")

    def _load_experiment_info(self) -> Optional[Dict[str, Any]]:
        """
        Load experiment information from the experiment directory if available.

        Returns:
            Experiment info dictionary or None if not available
        """
        if not self.experiment_dir or not os.path.isdir(self.experiment_dir):
            return None

        # Try to load training history
        history_path = os.path.join(
            self.experiment_dir, "results", "training_history.json"
        )
        if os.path.isfile(history_path):
            try:
                with open(history_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load training history: {e}")

        return None

    def _load_checkpoint(self) -> Dict[str, Any]:
        """Load checkpoint from file."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        print(f"Checkpoint loaded from epoch {checkpoint['epoch']}")
        print(f"Best validation loss: {checkpoint['best_val_loss']:.4f}")

        return checkpoint

    def _load_model(self) -> torch.nn.Module:
        """Load and initialize the model from checkpoint."""
        model_config = self.checkpoint["model_config"]
        model_type = model_config["model_type"]

        # Dynamically import and create the model based on type
        if model_type == "gru":
            from src.models.gru.model import GRUModel

            model = GRUModel(vocab_size=model_config["vocab_size"])

        elif model_type == "lstm":
            from src.models.lstm.model import LSTMModel

            model = LSTMModel(vocab_size=model_config["vocab_size"])

        elif model_type == "transformer":
            from src.models.transformer.model import TransformerModel

            model = TransformerModel(vocab_size=model_config["vocab_size"])

        elif model_type == "vae":
            from src.models.vae.model import VAE

            model = VAE(vocab_size=model_config["vocab_size"])

        elif model_type == "mamba":
            from src.models.mamba.model import MambaModel

            model = MambaModel(vocab_size=model_config["vocab_size"])

        elif model_type == "tcn":
            from src.models.tcn.model import TCNModel

            # TCN requires additional parameters
            embedding_dim = model_config.get("embedding_dim", 128)
            num_channels = model_config.get("num_channels", [100] * 8)
            model = TCNModel(
                vocab_size=model_config["vocab_size"],
                embedding_dim=embedding_dim,
                num_channels=num_channels,
            )

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Load state dict
        model.load_state_dict(self.checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def _prepare_initial_sequence(
        self, starter_text: Optional[str] = None
    ) -> torch.Tensor:
        """
        Prepare initial sequence for generation.

        Args:
            starter_text: Optional starter text (SMILES string)

        Returns:
            Initial sequence tensor
        """
        if starter_text is None:
            # Start with BOS token only
            initial_ids = [self.sampling_config["bos_token_id"]]
        else:
            # Tokenize the starter text WITHOUT special tokens to avoid EOS
            tokenized = self.tokenizer.tokenizer(
                starter_text,
                add_special_tokens=False,  # Don't add BOS/EOS automatically
                return_tensors="pt",
                truncation=True,
                padding=False,
            )
            starter_ids = tokenized["input_ids"].squeeze(0).tolist()

            # Manually prepend BOS token and avoid EOS token
            initial_ids = [self.sampling_config["bos_token_id"]] + starter_ids

            # Remove EOS token if it somehow got included
            eos_token_id = self.sampling_config["eos_token_id"]
            if initial_ids[-1] == eos_token_id:
                initial_ids = initial_ids[:-1]

        return torch.tensor([initial_ids], dtype=torch.long, device=self.device)

    def _apply_temperature(
        self, logits: torch.Tensor, temperature: float
    ) -> torch.Tensor:
        """Apply temperature scaling to logits."""
        if temperature == 0.0:
            temperature = 1e-8  # Avoid division by zero
        return logits / temperature

    def _top_k_filtering(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Apply top-k filtering to logits."""
        if top_k <= 0:
            return logits

        top_k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, top_k)
        min_values = values[:, -1, None]
        return torch.where(
            logits < min_values, torch.full_like(logits, -float("inf")), logits
        )

    def _top_p_filtering(self, logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Apply nucleus (top-p) filtering to logits."""
        if top_p >= 1.0:
            return logits

        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift right to keep at least one token
        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
        sorted_indices_to_remove[:, 0] = 0

        # Scatter sorted indices to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits = logits.masked_fill(indices_to_remove, -float("inf"))

        return logits

    def _apply_repetition_penalty(
        self, logits: torch.Tensor, input_ids: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Apply repetition penalty to logits."""
        if penalty == 1.0:
            return logits

        # Get unique tokens in the sequence
        unique_tokens = torch.unique(input_ids)

        # Apply penalty
        for token in unique_tokens:
            if token.item() < logits.size(-1):  # Valid token index
                if logits[0, token] > 0:
                    logits[0, token] = logits[0, token] / penalty
                else:
                    logits[0, token] = logits[0, token] * penalty

        return logits

    def _sample_next_token(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        generation_config: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Sample next token based on the configured strategy.

        Args:
            logits: Model logits for next token prediction
            input_ids: Current input sequence
            generation_config: Generation configuration

        Returns:
            Next token ID
        """
        # Apply repetition penalty
        if generation_config.get("repetition_penalty", 1.0) != 1.0:
            logits = self._apply_repetition_penalty(
                logits, input_ids, generation_config["repetition_penalty"]
            )

        # Apply temperature
        logits = self._apply_temperature(logits, generation_config["temperature"])

        # Apply filtering based on strategy
        strategy = generation_config["decoding_strategy"]

        if strategy == "greedy":
            # Greedy decoding - always pick the most likely token
            next_token = torch.argmax(logits, dim=-1)

        elif strategy == "top_k":
            # Top-k sampling
            logits = self._top_k_filtering(logits, generation_config["top_k"])
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        elif strategy in ["top_p", "nucleus"]:
            # Nucleus (top-p) sampling
            logits = self._top_p_filtering(logits, generation_config["top_p"])
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        elif strategy == "random":
            # Pure random sampling
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

        else:
            raise ValueError(f"Unsupported decoding strategy: {strategy}")

        return next_token

    def _standardize_model_output(
        self, outputs: Union[torch.Tensor, Tuple]
    ) -> torch.Tensor:
        """Standardize model outputs to get logits."""
        if isinstance(outputs, torch.Tensor):
            return outputs
        elif isinstance(outputs, tuple):
            # Return the first element (logits) for tuple outputs
            return outputs[0]
        else:
            raise ValueError(f"Unexpected output type: {type(outputs)}")

    def generate_single(
        self,
        starter_text: Optional[str] = None,
        max_length: Optional[int] = None,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """
        Generate a single molecular sequence.

        Args:
            starter_text: Optional starter text (SMILES string)
            max_length: Maximum length of generated sequence
            **generation_kwargs: Additional generation parameters

        Returns:
            Dictionary containing generated sequence and metadata
        """
        # Update generation config with any provided kwargs
        generation_config = self.sampling_config.copy()
        generation_config.update(generation_kwargs)

        if max_length is not None:
            generation_config["max_length"] = max_length

        # Prepare initial sequence
        input_ids = self._prepare_initial_sequence(starter_text)
        generated_ids = input_ids.clone()

        # Generation loop
        start_time = time.time()

        with torch.no_grad():
            for step in range(generation_config["max_length"]):
                # Forward pass
                outputs = self.model(input_ids)
                logits = self._standardize_model_output(outputs)

                # Get logits for next token (last position)
                next_token_logits = logits[:, -1, :]

                # Sample next token
                next_token = self._sample_next_token(
                    next_token_logits, input_ids, generation_config
                )

                # Add to sequence
                input_ids = torch.cat([input_ids, next_token], dim=-1)
                generated_ids = input_ids.clone()

                # Check for EOS token
                if (
                    generation_config.get("early_stopping", True)
                    and next_token.item() == generation_config["eos_token_id"]
                ):
                    break

                # Check minimum length
                if (
                    step + 1 >= generation_config.get("min_length", 0)
                    and next_token.item() == generation_config["eos_token_id"]
                ):
                    break

        generation_time = time.time() - start_time

        # Decode the sequence
        generated_sequence = generated_ids.squeeze(0).cpu().tolist()
        decoded_text = self.tokenizer.tokenizer.decode(
            generated_sequence,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        return {
            "generated_ids": generated_sequence,
            "decoded_text": decoded_text,
            "generation_time": generation_time,
            "sequence_length": len(generated_sequence),
            "starter_text": starter_text,
            "generation_config": generation_config,
        }

    def generate_filter_batch(
        self,
        logp_filter: Any,
        starter_text: Optional[str] = None,
        num_sequences: int = 5,
        max_length: Optional[int] = None,
        **generation_kwargs,
    ) -> Any:
        """ """
        counter = 0
        while counter < num_sequences:
            result = self.generate_single(
                starter_text=starter_text, max_length=max_length, **generation_kwargs
            )
            decoded_text = result["decoded_text"]
            is_valid, parts, logp_values = logp_filter.filter(sequence=decoded_text)
            if is_valid:
                result["sequence_id"] = counter
                if self.filtering_config.get("do_save_output", False):
                    result["smiles_parts"] = str(parts)
                    result["logp_values"] = str(logp_values)
                yield result
                counter += 1

    def generate_batch(
        self,
        starter_text: Optional[str] = None,
        num_sequences: int = 5,
        max_length: Optional[int] = None,
        **generation_kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate multiple molecular sequences.

        Args:
            starter_text: Optional starter text (single string or None for random start)
            num_sequences: Number of sequences to generate
            max_length: Maximum length of generated sequences
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generation results
        """
        results = []

        print(f"Generating {num_sequences} sequences...")
        if "discriminator_enabled" in self.filtering_config:
            from src.discriminator.logP_filter import LogPFilter

            logp_filter = LogPFilter(
                model_type=self.filtering_config.get(
                    "discriminator_model_type", "dmpnn"
                ),
                model_dir=self.filtering_config.get("discriminator_model_dir", "./"),
                logp_threshold=self.filtering_config.get(
                    "discriminator_logp_threshold", 1.0
                ),
            )
            for result in tqdm(
                self.generate_filter_batch(
                    logp_filter=logp_filter,
                    starter_text=starter_text,
                    num_sequences=num_sequences,
                    max_length=max_length,
                    **generation_kwargs,
                )
            ):
                results.append(result)
        else:
            for i in tqdm(range(num_sequences)):
                try:
                    result = self.generate_single(
                        starter_text=starter_text,
                        max_length=max_length,
                        **generation_kwargs,
                    )
                    result["sequence_id"] = i
                    results.append(result)
                except Exception as e:
                    print(f"Error generating sequence {i}: {e}")
                    continue

        return results

    def generate_with_preset(
        self, preset_name: str, num_sequences: int = 5, **generation_kwargs
    ) -> List[Dict[str, Any]]:
        """
        Generate sequences using predefined starter text presets.

        Args:
            preset_name: Name of the preset ('random', 'carbon_chain', 'aromatic', etc.)
            num_sequences: Number of sequences to generate
            **generation_kwargs: Additional generation parameters

        Returns:
            List of generation results
        """
        if preset_name not in STARTER_TEXTS:
            raise ValueError(
                f"Unknown preset: {preset_name}. Available: {list(STARTER_TEXTS.keys())}"
            )

        starter_config = STARTER_TEXTS[preset_name]

        if starter_config is None:
            # Random start
            starter_text = None
        elif isinstance(starter_config, str):
            # Single starter text
            starter_text = starter_config
        elif isinstance(starter_config, dict):
            # Pick the first option from dictionary
            starter_text = list(starter_config.values())[0]
        elif isinstance(starter_config, list):
            # Pick the first option from list
            starter_text = starter_config[0] if starter_config else None
        else:
            raise ValueError(f"Invalid starter configuration for preset: {preset_name}")

        return self.generate_batch(
            starter_text=starter_text, num_sequences=num_sequences, **generation_kwargs
        )

    def save_results(
        self,
        results: List[Dict[str, Any]],
        output_path: str,
        format: str = "json",
        use_experiment_dir: bool = True,
    ):
        """
        Save generation results to file.

        Args:
            results: List of generation results
            output_path: Path to save the results (relative to experiment dir if use_experiment_dir=True)
            format: Output format ('json', 'csv', 'txt')
            use_experiment_dir: If True and experiment directory is available,
                               save to experiment/results/samples/ directory
        """
        # Determine output path
        if (
            use_experiment_dir
            and self.experiment_dir
            and os.path.isdir(self.experiment_dir)
        ):
            # Save to experiment directory structure
            samples_dir = os.path.join(self.experiment_dir, "results", "samples")
            os.makedirs(samples_dir, exist_ok=True)

            # If output_path is just a filename, use it directly
            if os.path.dirname(output_path) == "":
                final_output_path = os.path.join(samples_dir, output_path)
            else:
                # If output_path contains directories, preserve the structure within samples
                final_output_path = os.path.join(samples_dir, output_path)
        else:
            # Use the path as-is
            final_output_path = output_path

        # Create directory only if the path contains a directory
        output_dir = os.path.dirname(final_output_path)
        if output_dir:  # Only create directory if it's not empty
            os.makedirs(output_dir, exist_ok=True)

        # Add sampling metadata to results
        enhanced_results = {
            "sampling_info": {
                "model_type": self.model_type,
                "checkpoint_path": self.checkpoint_path,
                "experiment_dir": self.experiment_dir,
                "generation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "sampling_config": self.sampling_config,
            },
            "results": results,
        }

        if format == "json":
            with open(final_output_path, "w") as f:
                json.dump(enhanced_results, f, indent=2)

        elif format == "csv":
            import pandas as pd

            df_data = []
            for result in results:
                data_to_append = {
                    "sequence_id": result.get("sequence_id", 0),
                    "decoded_text": result["decoded_text"],
                    "sequence_length": result["sequence_length"],
                    "generation_time": result["generation_time"],
                    "starter_text": result["starter_text"],
                }
                if self.filtering_config.get("do_save_output", False):
                    data_to_append = {
                        **data_to_append,
                        "smiles_parts": result.get("smiles_parts", ""),
                        "logp_values": result.get("logp_values", ""),
                    }
                    df_data.append(data_to_append)
                else:
                    df_data.append(data_to_append)

            df = pd.DataFrame(df_data)
            df.to_csv(final_output_path, index=False)

        elif format == "txt":
            with open(final_output_path, "w") as f:
                # Write header with sampling info
                f.write(f"Generated Sequences - {self.model_type.upper()} Model\n")
                f.write(f"Checkpoint: {os.path.basename(self.checkpoint_path)}\n")
                f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 60 + "\n\n")

                for i, result in enumerate(results):
                    f.write(f"Sequence {i + 1}:\n")
                    f.write(f"SMILES: {result['decoded_text']}\n")
                    f.write(f"Length: {result['sequence_length']}\n")
                    f.write(f"Time: {result['generation_time']:.3f}s\n")
                    if result.get("smiles_parts", False):
                        f.write(f"Parts: {result['smiles_parts']}\n")
                    if result.get("logp_values", False):
                        f.write(f"logP Values: {result['logp_values']}\n")
                    if result.get("starter_text", False):
                        f.write(f"Starter: {result['starter_text']}\n")
                    f.write("-" * 50 + "\n")
        else:
            raise ValueError(f"Unsupported format: {format}")

        print(f"Results saved to: {final_output_path}")
        if use_experiment_dir and self.experiment_dir:
            relative_path = os.path.relpath(final_output_path, self.experiment_dir)
            print(f"Relative to experiment: {relative_path}")

        return final_output_path

    def evaluate_sequences(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate generated sequences for basic statistics.

        Args:
            results: List of generation results

        Returns:
            Dictionary with evaluation metrics
        """
        if not results:
            return {}

        lengths = [r["sequence_length"] for r in results]
        times = [r["generation_time"] for r in results]

        evaluation = {
            "num_sequences": len(results),
            "avg_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": np.max(lengths),
            "avg_generation_time": np.mean(times),
            "total_time": np.sum(times),
            "unique_sequences": len(set(r["decoded_text"] for r in results)),
            "diversity_ratio": len(set(r["decoded_text"] for r in results))
            / len(results),
        }

        return evaluation


def create_sampler(
    checkpoint_path: str,
    device: Optional[str] = None,
    auto_detect_best: bool = True,
    sampling_kwargs: dict = {},
    filter_kwargs: dict = {},
    **kwargs,
) -> Sampler:
    """
    Factory function to create a molecular sampler.

    Args:
        checkpoint_path: Path to trained model checkpoint or experiment directory
        device: Device to use ('cuda', 'cpu', or None for auto-detect)
        auto_detect_best: If True, automatically find the best checkpoint in directories
        **sampling_kwargs: Additional sampling configuration
        **filter_kwargs: Additional filtering configuration

    Returns:
        Configured sampler instance
    """
    return Sampler(
        checkpoint_path=checkpoint_path,
        device=device,
        auto_detect_best=auto_detect_best,
        sampling_config=sampling_kwargs,
        filtering_config=filter_kwargs,
    )


if __name__ == "__main__":
    # Example usage with the new structured directory system
    print("🧪 MOLECULAR SAMPLER - Enhanced for Structured Experiments")
    print("=" * 60)

    # Example 1: Load from experiment directory (auto-detects best checkpoint)
    experiment_path = (
        "experiments/tcn_polymer_experiment_v1"  # Example experiment directory
    )

    # Use experiment directory directly (no fallback to old checkpoint structure)
    print(f"📁 Using experiment directory: {experiment_path}")

    try:
        # Create sampler with enhanced directory support
        sampler = create_sampler(
            checkpoint_path=experiment_path,
            temperature=0.8,
            max_length=50,
            auto_detect_best=True,  # Automatically find best checkpoint
        )

        print(f"\n✅ Sampler loaded successfully!")

        # Generate single sequence
        print(f"\n🔬 Generating single sequence...")
        result = sampler.generate_single(starter_text="C")
        print(f"Generated: {result['decoded_text']}")
        print(f"Length: {result['sequence_length']}")
        print(f"Time: {result['generation_time']:.3f}s")

        # Generate batch with same starter
        print(f"\n🧪 Generating batch of sequences with starter 'C'...")
        results = sampler.generate_batch(
            starter_text="C", num_sequences=5, max_length=30
        )

        # Generate batch without starter (random)
        print(f"\n🎲 Generating batch of random sequences...")
        random_results = sampler.generate_batch(
            starter_text=None, num_sequences=3, max_length=30
        )

        # Combine all results
        all_results = results + random_results

        # Evaluate results
        evaluation = sampler.evaluate_sequences(all_results)
        print(f"\n📊 Evaluation Results:")
        for key, value in evaluation.items():
            if isinstance(value, float):
                print(f"   {key}: {value:.3f}")
            else:
                print(f"   {key}: {value}")

        # Save results using structured directory system
        print(f"\n💾 Saving results...")

        # Save to experiment directory (should always be available with new structure)
        output_files = []
        if sampler.experiment_dir:
            # Save to structured experiment directory
            json_path = sampler.save_results(
                all_results, "generated_molecules.json", format="json"
            )
            csv_path = sampler.save_results(
                all_results, "generated_molecules.csv", format="csv"
            )
            txt_path = sampler.save_results(
                all_results, "generated_molecules.txt", format="txt"
            )
            output_files.extend([json_path, csv_path, txt_path])

            print(f"\n🎉 Molecular generation completed!")
            print(f"📁 Output files: {len(output_files)}")
            for file_path in output_files:
                print(f"   📄 {file_path}")

            print(f"\n💡 All outputs organized in experiment structure:")
            print(f"   📁 Experiment: {sampler.experiment_dir}")
            print(
                f"   📁 Samples: {os.path.join(sampler.experiment_dir, 'results', 'samples')}"
            )
        else:
            print(
                f"⚠️  Warning: No experiment directory found in checkpoint. This shouldn't happen with the new structure."
            )

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print(f"\n💡 Usage examples:")
        print(f"   📁 Experiment directory: 'experiments/tcn_polymer_experiment_v1'")
        print(
            f"   � Direct experiment path: 'tcn_polymer_experiment_v1' (searches in experiments/)"
        )
        print(
            f"\n🔧 Make sure you have run training with the new structured output system first!"
        )

    except Exception as e:
        print(f"❌ Unexpected error: {e}")

    print("=" * 60)
