"""
Advanced configuration management for different input types and model variations.
This module handles various input configurations for future extensibility.
"""

from typing import Dict, Any, List, Optional
import os


# Base data configurations for different input types
BASE_DATA_CONFIGS = {
    "smiles": {
        "target_column": "smiles",
        "data_path": "datasets/finals/dfs/master_df.csv",
        "test_size": 0.2,
        "random_state": 42,
        "input_type": "sequence",
        "tokenizer_type": "bert",
        "preprocessing": {
            "max_length": 512,
            "truncation": True,
            "padding": "max_length"
        }
    },
    
    "graph": {
        "target_column": "smiles",  # Will be converted to graph
        "data_path": "datasets/finals/dfs/master_df.csv",
        "test_size": 0.2,
        "random_state": 42,
        "input_type": "graph",
        "graph_featurizer": "rdkit",
        "preprocessing": {
            "node_features": ["atomic_number", "degree", "formal_charge", "hybridization"],
            "edge_features": ["bond_type", "is_aromatic", "is_in_ring"],
            "global_features": ["molecular_weight", "num_atoms", "num_bonds"]
        }
    },
    
    "fingerprint": {
        "target_column": "smiles",  # Will be converted to fingerprint
        "data_path": "datasets/finals/dfs/master_df.csv",
        "test_size": 0.2,
        "random_state": 42,
        "input_type": "fingerprint",
        "fingerprint_type": "morgan",
        "preprocessing": {
            "radius": 2,
            "n_bits": 2048,
            "use_features": True,
            "use_chirality": True
        }
    },
    
    "brics": {
        "target_column": "brics_fragments",
        "data_path": "datasets/brics_decompositions.txt",
        "test_size": 0.2,
        "random_state": 42,
        "input_type": "sequence",
        "tokenizer_type": "brics_specific",
        "preprocessing": {
            "max_fragments": 20,
            "fragment_separator": ".",
            "pad_fragments": True
        }
    }
}


# Model architecture configurations
MODEL_ARCHITECTURES = {
    "gru": {
        "base_config": {
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False
        },
        "variants": {
            "small": {
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_layers": 1,
                "dropout": 0.1
            },
            "large": {
                "embedding_dim": 512,
                "hidden_dim": 1024,
                "num_layers": 3,
                "dropout": 0.3
            },
            "bidirectional": {
                "embedding_dim": 256,
                "hidden_dim": 512,
                "num_layers": 2,
                "dropout": 0.2,
                "bidirectional": True
            }
        },
        "supported_inputs": ["smiles", "brics"],
        "task_types": ["generation", "classification", "regression"]
    },
    
    "lstm": {
        "base_config": {
            "embedding_dim": 256,
            "hidden_dim": 512,
            "num_layers": 2,
            "dropout": 0.2,
            "bidirectional": False
        },
        "variants": {
            "small": {
                "embedding_dim": 128,
                "hidden_dim": 256,
                "num_layers": 1,
                "dropout": 0.1
            },
            "large": {
                "embedding_dim": 512,
                "hidden_dim": 1024,
                "num_layers": 3,
                "dropout": 0.3
            },
            "attention": {
                "embedding_dim": 256,
                "hidden_dim": 512,
                "num_layers": 2,
                "dropout": 0.2,
                "use_attention": True,
                "attention_dim": 128
            }
        },
        "supported_inputs": ["smiles", "brics"],
        "task_types": ["generation", "classification", "regression"]
    },
    
    "transformer": {
        "base_config": {
            "embedding_dim": 256,
            "n_head": 8,
            "n_layers": 6,
            "dropout": 0.1,
            "dim_feedforward": 1024,
            "aggregation_type": "mean"
        },
        "variants": {
            "small": {
                "embedding_dim": 128,
                "n_head": 4,
                "n_layers": 3,
                "dropout": 0.1,
                "dim_feedforward": 512
            },
            "large": {
                "embedding_dim": 512,
                "n_head": 16,
                "n_layers": 12,
                "dropout": 0.1,
                "dim_feedforward": 2048
            },
            "gpt_style": {
                "embedding_dim": 256,
                "n_head": 8,
                "n_layers": 6,
                "dropout": 0.1,
                "dim_feedforward": 1024,
                "causal_mask": True,
                "task_type": "generation"
            }
        },
        "supported_inputs": ["smiles", "brics"],
        "task_types": ["generation", "classification", "regression"]
    },
    
    "vae": {
        "base_config": {
            "embedding_dim": 256,
            "hidden_dim": 512,
            "z_dim": 128,
            "num_layers": 2,
            "dropout": 0.2,
            "encoder_type": "gru",
            "decoder_type": "gru"
        },
        "variants": {
            "small": {
                "embedding_dim": 128,
                "hidden_dim": 256,
                "z_dim": 64,
                "num_layers": 1,
                "dropout": 0.1
            },
            "large": {
                "embedding_dim": 512,
                "hidden_dim": 1024,
                "z_dim": 256,
                "num_layers": 3,
                "dropout": 0.3
            },
            "beta_vae": {
                "embedding_dim": 256,
                "hidden_dim": 512,
                "z_dim": 128,
                "num_layers": 2,
                "dropout": 0.2,
                "beta": 4.0  # Higher beta for disentanglement
            }
        },
        "supported_inputs": ["smiles", "brics"],
        "task_types": ["generation", "representation_learning"]
    },
    
    "graph_transformer": {
        "base_config": {
            "hidden_dims": {
                "dx": 256,
                "de": 128,
                "dy": 64,
                "n_head": 8,
                "dim_ffX": 1024,
                "dim_ffE": 512
            },
            "n_layers": 6,
            "dropout": 0.1
        },
        "variants": {
            "small": {
                "hidden_dims": {
                    "dx": 128,
                    "de": 64,
                    "dy": 32,
                    "n_head": 4,
                    "dim_ffX": 512,
                    "dim_ffE": 256
                },
                "n_layers": 3
            },
            "large": {
                "hidden_dims": {
                    "dx": 512,
                    "de": 256,
                    "dy": 128,
                    "n_head": 16,
                    "dim_ffX": 2048,
                    "dim_ffE": 1024
                },
                "n_layers": 12
            }
        },
        "supported_inputs": ["graph"],
        "task_types": ["generation", "classification", "regression"]
    },
    
    "mamba": {
        "base_config": {
            "embedding_dim": 256,
            "state_size": 16,
            "conv_kernel": 4,
            "expand_factor": 2,
            "num_layers": 6,
            "dropout": 0.1
        },
        "variants": {
            "small": {
                "embedding_dim": 128,
                "state_size": 8,
                "conv_kernel": 3,
                "expand_factor": 2,
                "num_layers": 3,
                "dropout": 0.1
            },
            "large": {
                "embedding_dim": 512,
                "state_size": 32,
                "conv_kernel": 6,
                "expand_factor": 4,
                "num_layers": 12,
                "dropout": 0.1
            }
        },
        "supported_inputs": ["smiles", "brics"],
        "task_types": ["generation", "classification", "regression"]
    }
}


# Training configurations for different scenarios
TRAINING_CONFIGURATIONS = {
    "quick_test": {
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "patience": 2,
        "grad_clip": 1.0
    },
    
    "development": {
        "epochs": 10,
        "batch_size": 32,
        "learning_rate": 1e-3,
        "patience": 5,
        "grad_clip": 1.0
    },
    
    "production": {
        "epochs": 50,
        "batch_size": 64,
        "learning_rate": 1e-4,
        "patience": 10,
        "grad_clip": 0.5,
        "scheduler": "cosine",
        "warmup_steps": 1000
    },
    
    "fine_tuning": {
        "epochs": 20,
        "batch_size": 32,
        "learning_rate": 1e-5,
        "patience": 7,
        "grad_clip": 0.1,
        "scheduler": "step",
        "step_size": 5,
        "gamma": 0.5
    }
}


# Task-specific configurations
TASK_CONFIGURATIONS = {
    "molecule_generation": {
        "loss_function": "cross_entropy",
        "metrics": ["perplexity", "validity", "uniqueness", "novelty"],
        "evaluation": {
            "sample_size": 1000,
            "temperature": 1.0,
            "top_k": 50
        }
    },
    
    "property_prediction": {
        "loss_function": "mse",
        "metrics": ["mae", "rmse", "r2"],
        "evaluation": {
            "cross_validation": True,
            "cv_folds": 5
        }
    },
    
    "classification": {
        "loss_function": "cross_entropy",
        "metrics": ["accuracy", "precision", "recall", "f1"],
        "evaluation": {
            "class_weights": "balanced"
        }
    }
}


class ConfigManager:
    """
    Centralized configuration manager for handling different input types and model variations.
    """
    
    def __init__(self):
        self.data_configs = BASE_DATA_CONFIGS
        self.model_architectures = MODEL_ARCHITECTURES
        self.training_configs = TRAINING_CONFIGURATIONS
        self.task_configs = TASK_CONFIGURATIONS
    
    def get_data_config(self, input_type: str, custom_overrides: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Get data configuration for a specific input type.
        
        Args:
            input_type: Type of input data ('smiles', 'graph', 'fingerprint', 'brics')
            custom_overrides: Custom configuration overrides
            
        Returns:
            Data configuration dictionary
        """
        if input_type not in self.data_configs:
            raise ValueError(f"Unsupported input type: {input_type}")
        
        config = self.data_configs[input_type].copy()
        
        if custom_overrides:
            config.update(custom_overrides)
        
        return config
    
    def get_model_config(
        self, 
        model_name: str, 
        variant: str = "base", 
        custom_overrides: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get model configuration for a specific architecture and variant.
        
        Args:
            model_name: Name of the model architecture
            variant: Model variant ('base', 'small', 'large', etc.)
            custom_overrides: Custom configuration overrides
            
        Returns:
            Model configuration dictionary
        """
        if model_name not in self.model_architectures:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model_arch = self.model_architectures[model_name]
        
        if variant == "base":
            config = model_arch["base_config"].copy()
        elif variant in model_arch.get("variants", {}):
            # Start with base config and override with variant
            config = model_arch["base_config"].copy()
            config.update(model_arch["variants"][variant])
        else:
            raise ValueError(f"Unsupported variant '{variant}' for model '{model_name}'")
        
        if custom_overrides:
            config.update(custom_overrides)
        
        return config
    
    def get_training_config(
        self, 
        scenario: str = "development", 
        custom_overrides: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Get training configuration for a specific scenario.
        
        Args:
            scenario: Training scenario ('quick_test', 'development', 'production', 'fine_tuning')
            custom_overrides: Custom configuration overrides
            
        Returns:
            Training configuration dictionary
        """
        if scenario not in self.training_configs:
            raise ValueError(f"Unsupported training scenario: {scenario}")
        
        config = self.training_configs[scenario].copy()
        
        if custom_overrides:
            config.update(custom_overrides)
        
        return config
    
    def get_task_config(self, task_type: str) -> Dict[str, Any]:
        """
        Get task-specific configuration.
        
        Args:
            task_type: Type of task ('molecule_generation', 'property_prediction', 'classification')
            
        Returns:
            Task configuration dictionary
        """
        if task_type not in self.task_configs:
            raise ValueError(f"Unsupported task type: {task_type}")
        
        return self.task_configs[task_type].copy()
    
    def validate_combination(self, model_name: str, input_type: str, task_type: str) -> bool:
        """
        Validate if a combination of model, input type, and task is supported.
        
        Args:
            model_name: Name of the model architecture
            input_type: Type of input data
            task_type: Type of task
            
        Returns:
            True if combination is valid, False otherwise
        """
        if model_name not in self.model_architectures:
            return False
        
        model_info = self.model_architectures[model_name]
        
        # Check if input type is supported
        if input_type not in model_info.get("supported_inputs", []):
            return False
        
        # Check if task type is supported
        if task_type not in model_info.get("task_types", []):
            return False
        
        return True
    
    def get_recommended_config(
        self, 
        model_name: str, 
        input_type: str, 
        task_type: str,
        scenario: str = "development"
    ) -> Dict[str, Any]:
        """
        Get a complete recommended configuration for a specific setup.
        
        Args:
            model_name: Name of the model architecture
            input_type: Type of input data
            task_type: Type of task
            scenario: Training scenario
            
        Returns:
            Complete configuration dictionary
        """
        if not self.validate_combination(model_name, input_type, task_type):
            raise ValueError(f"Invalid combination: {model_name} + {input_type} + {task_type}")
        
        config = {
            "model_config": self.get_model_config(model_name),
            "data_config": self.get_data_config(input_type),
            "training_config": self.get_training_config(scenario),
            "task_config": self.get_task_config(task_type),
            "metadata": {
                "model_name": model_name,
                "input_type": input_type,
                "task_type": task_type,
                "scenario": scenario
            }
        }
        
        return config
    
    def list_available_options(self) -> Dict[str, List[str]]:
        """
        List all available configuration options.
        
        Returns:
            Dictionary with lists of available options
        """
        return {
            "input_types": list(self.data_configs.keys()),
            "model_architectures": list(self.model_architectures.keys()),
            "training_scenarios": list(self.training_configs.keys()),
            "task_types": list(self.task_configs.keys())
        }


# Global configuration manager instance
config_manager = ConfigManager()


def get_config_for_combination(
    model_name: str,
    input_type: str = "smiles",
    task_type: str = "molecule_generation",
    scenario: str = "development",
    model_variant: str = "base"
) -> Dict[str, Any]:
    """
    Convenience function to get complete configuration for a specific combination.
    
    Args:
        model_name: Name of the model architecture
        input_type: Type of input data
        task_type: Type of task
        scenario: Training scenario
        model_variant: Model variant
        
    Returns:
        Complete configuration dictionary
    """
    config = config_manager.get_recommended_config(model_name, input_type, task_type, scenario)
    
    # Update model config with variant
    if model_variant != "base":
        variant_config = config_manager.get_model_config(model_name, model_variant)
        config["model_config"].update(variant_config)
    
    return config
