# Sampling Configuration

# Default sampling configuration
DEFAULT_SAMPLING_CONFIG = {
    # Generation parameters
    'max_length': 100,
    'temperature': 1.0,
    'top_k': 50,
    'top_p': 0.9,
    'do_sample': True,
    'num_return_sequences': 1,
    
    # Decoding strategy
    'decoding_strategy': 'top_p',  # 'greedy', 'beam', 'top_k', 'top_p', 'nucleus'
    'beam_size': 5,  # For beam search
    
    # Special tokens
    'pad_token_id': 0,
    'eos_token_id': 102,  # BERT [SEP] token
    'bos_token_id': 101,  # BERT [CLS] token
    'unk_token_id': 100,  # BERT [UNK] token
    
    # Sampling control
    'repetition_penalty': 1.0,
    'length_penalty': 1.0,
    'no_repeat_ngram_size': 3,
    
    # Early stopping
    'early_stopping': True,
    'min_length': 5,
}

# Predefined starter texts for different types of molecules
STARTER_TEXTS = {
    'random': None,  # Start with BOS token only
    'carbon_chain': 'C',
    'aromatic': 'c1ccccc1',
    'polymer_unit': 'CC',
    'functional_groups': {
        'alcohol': 'CCO',
        'carboxylic_acid': 'CC(=O)O',
        'amine': 'CCN',
        'ester': 'CC(=O)OC',
        'ether': 'COC',
        'aldehyde': 'CC=O',
        'ketone': 'CC(=O)C',
    },
    'ring_systems': {
        'benzene': 'c1ccccc1',
        'cyclohexane': 'C1CCCCC1',
        'pyridine': 'c1ccncc1',
        'furan': 'c1ccoc1',
    },
    'custom': []  # User-defined starter texts
}

# Model-specific sampling adjustments
MODEL_SAMPLING_CONFIGS = {
    'transformer': {
        'temperature': 0.8,
        'top_p': 0.9,
        'repetition_penalty': 1.1,
    },
    'gru': {
        'temperature': 1.0,
        'top_k': 40,
        'repetition_penalty': 1.0,
    },
    'lstm': {
        'temperature': 1.0,
        'top_k': 40,
        'repetition_penalty': 1.0,
    },
    'vae': {
        'temperature': 1.2,
        'top_p': 0.95,
        'repetition_penalty': 0.9,
    },
    'mamba': {
        'temperature': 0.9,
        'top_p': 0.92,
        'repetition_penalty': 1.05,
    },
    'tcn': {
        'temperature': 1.0,
        'top_k': 45,
        'repetition_penalty': 1.0,
    },
}
