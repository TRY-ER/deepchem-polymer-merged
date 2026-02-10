# SMiPoly Polymer Generator (polg.py) - Understanding the biplym Function

## Overview

The `biplym` function in `polg.py` is a core component of the SMiPoly (SMILES-based Polymer) system that generates polymers from classified monomers. It takes a DataFrame of monomer candidates and produces polymer structures based on predefined polymerization rules.

## What the biplym Function Does

### Primary Purpose
The `biplym` function generates polymers by:
1. **Processing monomer data** from an input DataFrame containing classified monomers
2. **Applying polymerization rules** based on target polymer classes
3. **Creating polymer structures** using RDKit chemistry operations
4. **Filtering and deduplicating** results to produce a clean output

### Key Steps in the Process

#### 1. Data Loading and Preparation (Lines 32-52)
The function loads several rule files from the `rules` directory:
- `mon_vals.json`: Monomer classification indices
- `mon_dic.json`: Monomer type dictionary (e.g., "vinyl": 1, "epo": 2)
- `mon_lst.json`: Monomer lists
- `excl_lst.json`: Exclusion lists
- `ps_rxn.pkl`: Polymerization reaction rules (pickle format)
- `ps_class.json`: Polymer class definitions
- `ps_gen.pkl`: Polymer generation rules

#### 2. Input Validation and Target Selection (Lines 93-119)
- Sets default parameters if not provided
- Validates target polymer classes against available options
- Supports special targets like `['all',]` or `['exc_ole',]` (exclude polyolefins)

#### 3. DataFrame Processing (Lines 121-138)
- Removes unnecessary columns (like 'ROMol')
- Filters for valid monomer candidates
- Converts boolean columns from string representations
- Selects relevant columns based on monomer dictionary

#### 4. Polymer Generation Loop (Lines 144-221)
The core generation process iterates through:
- **Target polymer classes** (e.g., polyester, polyamide, polyurethane)
- **Polymerization sets** within each class
- **Monomer combinations** that match the polymerization rules

For each combination:
- **Bipolymerization**: When two different monomers are involved
- **Homopolymerization**: When a single monomer self-polymerizes
- Applies RDKit reactions to generate polymer structures

#### 5. Post-Processing (Lines 223-244)
- Explodes polymer lists to individual entries
- Removes invalid/empty polymer entries
- Eliminates duplicate reactions
- Sorts monomer pairs to identify unique reaction sets
- Returns final DataFrame with generated polymers

## Data Sources and Structure

### Monomer Classifications
The system uses a hierarchical classification system:

**Primary Monomer Types** (from `mon_vals.json`):
- **Group 0** [1-13]: Basic monomers (vinyl, epo, lactone, etc.)
- **Group 1** [51-58]: Di-functional monomers (diepo, diCOOH, diol, etc.)
- **Group 2** [200-206]: Specialized monomers
- **Group 3** [1001+]: Olefinic monomers (acryl, styryl, allyl, etc.)

### Polymer Classes
Available polymer classes (from `ps_class.json`):
- `polyolefin`: Olefin-based polymers
- `polyester`: Ester-linked polymers
- `polyether`: Ether-linked polymers
- `polyamide`: Amide-linked polymers
- `polyimide`: Imide-based polymers
- `polyurethane`: Urethane-linked polymers
- `polyoxazolidone`: Oxazolidone-based polymers

### Input DataFrame Requirements
The input DataFrame must contain:
- `smip_cand_mons`: SMILES strings of candidate monomers
- Boolean columns indicating monomer types (e.g., 'vinyl', 'epo', 'diol')
- Optional: `ROMol` column (automatically removed)

### Output DataFrame Structure
The function returns a DataFrame with:
- `mon1`: First monomer SMILES
- `mon2`: Second monomer SMILES (empty for homopolymers)
- `polym`: Generated polymer structure
- `polymer_class`: Classification of the polymer
- `Ps_rxnL`: Reaction key/identifier
- `Ps_rxn_smarts`: SMARTS representation of the reaction

## Key Functions Used

### From funclib.py
- `genmol()`: Converts SMILES to RDKit molecule objects
- `bipolymA()`: Performs bipolymerization reactions
- `homopolymA()`: Performs homopolymerization reactions

### RDKit Operations
- `AllChem.ReactionToSmarts()`: Converts reactions to SMARTS format
- Molecular substructure matching and manipulation

## Performance Considerations

### Multithreaded Version
The file also includes `biplym_mt()` (lines 528-683), a multithreaded version that:
- Uses process pools for parallel execution
- Implements data-parallel chunking
- Optimizes memory usage for large datasets
- Provides significant speedup for batch processing

### Memory Management
- Filters data early to reduce processing load
- Uses chunking strategies for large monomer sets
- Implements efficient deduplication algorithms

## Usage Examples

```python
# Basic usage - generate all polymer types
result_df = biplym(input_monomer_df)

# Generate specific polymer classes
result_df = biplym(input_monomer_df, targ=['polyester', 'polyamide'])

# Generate all except polyolefins
result_df = biplym(input_monomer_df, targ=['exc_ole'])

# Display processing statistics
result_df = biplym(input_monomer_df, dsp_rsl=True)

# Use multithreaded version for large datasets
result_df = biplym_mt(input_monomer_df, max_workers=8)
```

## Error Handling

The function includes comprehensive error handling:
- Validates target polymer classes
- Handles missing or invalid monomer data
- Manages RDKit conversion failures gracefully
- Provides informative error messages for debugging

## Integration with SMiPoly System

The `biplym` function is part of a larger polymer generation pipeline:
1. **Monomer classification** identifies reactive functional groups
2. **Polymer generation** (biplym) applies polymerization rules
3. **Property prediction** can be applied to generated polymers
4. **Database storage** for downstream analysis

This system enables high-throughput virtual polymer screening and discovery for materials science applications.