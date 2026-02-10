from rdkit import Chem
from rdkit.Chem import AllChem

smiles = "OCCOCCOCCOCCOCCOC(C)C(=O)OC(C)C(=O)OC(C)C(=O)OC(C)C(=O)OC(C)C(=O)"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)

# Write out as PDB instead of SDF
# Use RDKit helper to write a PDB file from the molecule's conformer
Chem.MolToPDBFile(mol, 'peo_pla_initial.pdb')
