from rdkit import Chem
from rdkit.Chem import rdChemReactions

class StringValidator:
    """
    Example -> [5*]NCN[5*].*Nc1ccc(NCCC[4*])cc1|[4*][*:1].[5*][*:2]>>[*:1][*:2] |[4*]-[*:1].[5*]-[*:2]>>[$([C&!D1&!$(C=*)]-&!@[#6]):1]-&!@[$([N&!D1&!$(N=*)&!$(N-[!#6&!#16&!#0&!#1])&!$([N&R]@[C&R]=O)]):2]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5 

    Part 1 -> SMILES ->  [5*]NCN[5*].*Nc1ccc(NCCC[4*])cc1
    Part 2 -> Reaction SMARTS ->  [4*]-[*:1].[5*]-[*:2]>>[$([C&!D1&!$(C=*)]-&!@[#6]):1]-&!@[$([N&!D1&!$(N=*)&!$(N-[!#6&!#16&!#0&!#1])&!$([N&R]@[C&R]=O)]):2]
    Part 3 -> Weight Distribution -> |0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5

    Validation Layer:

    1. Grammar Validation
        - Check for valid SMILES, SMARTS syntax
        - Ensure correct use of atom and bond symbols
        - Ensure correct weight variation is assigned  
    2. Syntax Validation
        - Ensure right indexing from the reaction components
        - Check for correct use of brackets and parentheses and transformations
        - Check for placement of "|" in the final representation
    """

    def __init__(self, weight_variations=None):
        if weight_variations is None:
            weight_variations = [
                "|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5",
                "|0.5|0.5|<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.125:0.125<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125"
            ]
        self.weight_variations = weight_variations

    @staticmethod
    def parse_parts(value: str):
        split_values = value.split("|")
        smiles_part = split_values[0]
        smarts_part = split_values[1]
        weight_part = "|" + "|".join(split_values[2:])
        return smiles_part, smarts_part, weight_part

    def _validate_smiles_grammar_candidate(self, str_part: str) -> None:
        try:
            Chem.MolFromSmiles(str_part)
        except Exception as e:
            print("Invalid SMILES:", str_part)
            raise e

    def _validate_smarts_reaction(self, str_part: str) -> None:
        rxn = rdChemReactions.ReactionFromSmarts(str_part) 
        if rxn is None:
            raise ValueError("Invalid Reaction SMARTS")

    def _validate_grammar(self, candidate: str) -> None:
        smiles_section, smarts_reaction_section, weight_val = StringValidator.parse_parts(candidate)
        if weight_val not in self.weight_variations:
            raise ValueError("Invalid Weight Variation")
        self._validate_smiles_grammar_candidate(smiles_section)
        self._validate_smarts_reaction(smarts_reaction_section)

    def _validate_syntax(self, candidate: str) -> None:
        ...

    def _validate_mapping(self, candidate: str) -> None:
        ...

    def validate(self, candidate: str):
        try:
            self._validate_grammar(candidate)
            self._validate_syntax(candidate)
            self._validate_mapping(candidate)
            return True
        except Exception as e:
            print("Validation Error:", e)
            return False
            
if __name__ == "__main__":
    validator = StringValidator()
    validator.validate("[5*]NCN[5*].[1*]Nc1ccc(NCCC[4*])cc1|[4*]-[*:1].[5*]-[*:2]>>[$([C&!D1&!$(C=*)]-&!@[#6]):1]-&!@[$([N&!D1&!$(N=*)&!$(N-[!#6&!#16&!#0&!#1])&!$([N&R]@[C&R]=O)]):2]|0.5|0.5|<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.125:0.125<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125")
    