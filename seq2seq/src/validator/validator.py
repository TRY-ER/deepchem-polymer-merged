import re

from rdkit import Chem
from rdkit.Chem import AllChem

reaction_smarts_map = {
    "E": [
        "([O&X2&H1&!$(OC=*):1].[C&X3:2](=O)[O&X2&H1])>>(*-[O&X2:1].[C&X3:2](=O)-*)",
        "[C&X3&R:1](=[O&X1])[O&X2&R:2]>>(*-[C&X3:1]=[O&X1].[O&X2:2]-*)",
        "([O&X2&H1&!$(OC=*):1].[C&X3:2](=O)[O&X2&H1]).([O&X2&H1&!$(OC=*):3].[C&X3:4](=O)[O&X2&H1])>>(*-[O&X2:1].[C&X3:2](=O)[O&X2:3].[C&X3:4](=O)-*)",
        "([O,S;X2;H1;!$([O,S]C=*):1].[O,S;X2;H1;!$([O,S]C=*):2]).[C&-]#[O&+]>>(*-[O,S;X2;!$([O,S]C=*):1].[O,S;X2;!$([O,S]C=*):2][C&X3](=O)-*)",
        "([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*)",
        "[C,c;R:1][C&X3,c;R](=[O&X1])[O&X2,o;R][C&X3,c;R](=[O&X1])[C,c;R:2].[C&X4&R:3]1[O&X2&R:4][C&X4&R:5]1>>([C,c:1][C&X3](=[O&X1])-*.[C,c:2][C&X3](=[O&X1])[O&X2][C&X4:3][C&X4:5][O&X2:4]-*)",
    ],
    "Et": [
        "[C&X4;H2,H1,H0;R:1]1[O&R][C&R:2]1>>*-[C&X4:1][C&X4:2]O-*",
        "c1([O&H1:1])[c:2][c:3][c&H1:4][c:5][c:6]1>>c1([O&X2:1]-*)[c:2][c:3][c:4](-*)[c:5][c:6]1",
        "[c:1]1[c:2][c:3]([F,Cl,Br,I])[c:4][c:5][c:6]1[S&X4](=[O&X1])(=[O&X1])[c:7]1[c:8][c:9][c:10]([F,Cl,Br,I])[c:11][c:12]1.([O&X2&H1&!$([O,S]C=*):13].[O&X2&H1&!$([O,S]C=*):14])>>([c:1]1[c:2][c:3](-*)[c:4][c:5][c:6]1[S&X4](=[O&X1])(=[O&X1])[c:7]1[c:8][c:9][c:10]([O&X2&!$([O,S]C=*):13])[c:11][c:12]1.[O&X2&!$([O,S]C=*):14]-*)",
        "[c:1]1[c:2][c:3](F)[c:4][c:5][c:6]1[C&X3](=[O&X1])[c:7]1[c:8][c:9][c:10](F)[c:11][c:12]1.([O&X2&H1&!$([O,S]C=*):13].[O&X2&H1&!$([O,S]C=*):14])>>([c:1]1[c:2][c:3](-*)[c:4][c:5][c:6]1[C&X3](=[O&X1])[c:7]1[c:8][c:9][c:10]([O&X2&!$([O,S]C=*):13])[c:11][c:12]1.[O&X2&!$([O,S]C=*):14]-*)",
    ],
    "A": [
        "[C&X3&R:1](=[O&X1])[N&X3&R:2]>>(*-[C&X3:1]=[O&X1].[N&X3:2]-*)",
        "([N&X3;H2,H1;!$(OC=*):1].[C&X3:2](=O)[O&X2&H1])>>(*-[N&X3:1].[C&X3:2](=O)-*)",
        "([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([N&X3;H2,H1;!$(NC=*):3].[N&X3;H2,H1;!$(NC=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[N&X3&!$(NC=*):3].[N&X3&!$(NC=*):4]-*)",
        "([N&X3;H2,H1;!$(NC=*):1].[C&X3:2](=O)[O&X2&H1]).([N&X3;H2,H1;!$(NC=*):3].[C&X3:4](=O)[O&X2&H1])>>(*-[N&X3&!$(NC=*):1].[C&X3:2](=O)[N&X3&!$(NC=*):3].[C&X3:4](=O)-*)",
    ],
    "I": [
        "([C&X3,c;R:1](=[O&X1])[O&X2,o;R][C&X3,c;R:2]=[O&X1].[C&X3,c;R:3](=[O&X1])[O&X2,o;R][C&X3,c;R:4]=[O&X1]).([C,c:5][N&X3&H2&!$(N[C,S]=*)].[C,c:6][N&X3&H2&!$(N[C,S]=*)])>>([C&X3,c;R:1](=[O&X1])[N&X3&R]([C,c:5])[C&X3,c;R:2]=[O&X1].[C,c:6]-*.[C&X3,c;R:3](=[O&X1])[N&X3&R](-*)[C&X3&R:4]=[O&X1])"
    ],
    "U": [
        "([N&X2:1]=[C&X2]=[O&X1,S&X1:2].[N&X2:3]=[C&X2:4]=[O&X1,S&X1:5]).([O&X2,S&X2;H1;!$([O,S]C=*):6].[O&X2,S&X2;H1;!$([O,S]C=*):7])>>(*-[C&X3](=[O&X1,S&X1:2])[N&X3:1].[N&X3:3][C&X3:4](=[O&X1,S&X1:5])[O&X2,S&X2;!$([O,S]C=*):6].[O&X2,S&X2;!$([O,S]C=*):7]-*)"
    ],
}


# some utility functions
def _get_nonaromatic_nitrogen(smiles_string):
    if not isinstance(smiles_string, str):
        return 0
    # Pattern to find 'N' that is a nitrogen atom, not part of a two-letter element symbol.
    # Excludes 'Na', 'Nb', 'Nd', 'Ne', 'Ni', 'Np'.
    pattern = r"N(?![abdeip])"
    matches = re.findall(pattern, smiles_string)
    return len(matches)


def _get_nonaromatic_oxygen(smiles_string):
    if not isinstance(smiles_string, str):
        return 0
    # Pattern to find 'O' that is an oxygen atom, not part of a two-letter element symbol.
    # Excludes 'Os', 'Og'.
    pattern = r"O(?![sg])"
    matches = re.findall(pattern, smiles_string)
    return len(matches)


def _get_num_rings(smiles_string):
    if not isinstance(smiles_string, str):
        return 0
    # Count all digits in the SMILES string
    digits = re.findall(r"\d", smiles_string)
    # Each ring is typically denoted by two identical digits (e.g., C1CCCCC1), so divide by 2
    return len(digits) // 2


class StringValidatorBRICSBase:
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
                "|0.5|0.5|<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.125:0.125<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125",
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
        rxn = AllChem.ReactionFromSmarts(str_part)
        if rxn is None:
            raise ValueError("Invalid Reaction SMARTS")

    def _validate_grammar(self, candidate: str) -> None:
        smiles_section, smarts_reaction_section, weight_val = (
            StringValidator.parse_parts(candidate)
        )
        if weight_val not in self.weight_variations:
            raise ValueError("Invalid Weight Variation")
        self._validate_smiles_grammar_candidate(smiles_section)
        self._validate_smarts_reaction(smarts_reaction_section)

    def _validate_syntax(self, candidate: str) -> None: ...

    def _validate_mapping(self, candidate: str) -> None: ...

    def validate(self, candidate: str):
        try:
            self._validate_grammar(candidate)
            self._validate_syntax(candidate)
            self._validate_mapping(candidate)
            return True
        except Exception as e:
            print("Validation Error:", e)
            return False


class SeqValidator:
    """
    Variation 0
    -----------
    Homo_Candidate = ##E:0:4:1:27## ${[[6*]C(=O)O] + [[16*]c1cc(C)cc(C(=O)O)c1O] -> [6*]-[*:1].[16*]-[*:2]>>[$([C&D3&!R](=O)-&!@[#0,#6,#7,#8]):1]-&!@[$([c&$(c(:c):c)]):2] -> [Cc1cc(C(=O)O)c(O)c(C(=O)O)c1]} => ([O&X2&H1&!$(OC=*):1].[C&X3:2](=O)[O&X2&H1])>>(*-[O&X2:1].[C&X3:2](=O)-*) => *Oc1c(C(*)=O)cc(C)cc1C(=O)O$
    Co_Candidate = ##E:0:8:6:83## ${[[6*]C(=O)Cl] + [[16*]c1cc(C)c(C)cc1CC(=O)O] -> [6*]-[*:1].[16*]-[*:2]>>[$([C&D3&!R](=O)-&!@[#0,#6,#7,#8]):1]-&!@[$([c&$(c(:c):c)]):2] -> [Cc1cc(CC(=O)O)c(C(=O)Cl)cc1C]} + {[[16*]c1ccc(O)cc1O] + [[16*]c1cc2c(cc1-c1cc(O)cc(O)c1)C(=O)c1ccccc1C2=O] -> [16*]-[*:1].[16*]-[*:2]>>[$([c&$(c(:c):c)]):1]-&!@[$([c&$(c(:c):c)]):2] -> [O=C1c2ccccc2C(=O)c2cc(-c3ccc(O)cc3O)c(-c3cc(O)cc(O)c3)cc21]} => ([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*) => *Oc1cc(O)cc(-c2cc3c(cc2-c2ccc(O)cc2OC(=O)Cc2cc(C)c(C)cc2C(*)=O)C(=O)c2ccccc2C3=O)c1$


    Variation 1
    -----------
    Homo_Candidate =  ##E:0:4:1:27## ${Cc1cc(C(=O)O)c(O)c(C(=O)O)c1} => ([O&X2&H1&!$(OC=*):1].[C&X3:2](=O)[O&X2&H1])>>(*-[O&X2:1].[C&X3:2](=O)-*) => *Oc1c(C(*)=O)cc(C)cc1C(=O)O$
    Co_Candidate = ##E:0:8:6:83## ${Cc1cc(CC(=O)O)c(C(=O)Cl)cc1C} + {O=C1c2ccccc2C(=O)c2cc(-c3ccc(O)cc3O)c(-c3cc(O)cc(O)c3)cc21} => ([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*) => *Oc1cc(O)cc(-c2cc3c(cc2-c2ccc(O)cc2OC(=O)Cc2cc(C)c(C)cc2C(*)=O)C(=O)c2ccccc2C3=O)c1$

    """

    def __init__(self, variation_code=1):
        assert variation_code in [0, 1], "Invalid variation code"
        self.variation_code = variation_code

    def _parse_expression_token(self, expression_token):
        token = expression_token.strip("$")
        portions = token.split("=>")
        assert len(portions) == 3, "<invalid-parse-expression>"
        reactants, reaction, product = portions
        reactants = [reactant.strip()[1:-1] for reactant in reactants.split("+")]
        return reactants, reaction, product

    def _parse(self, sequence):
        # init token is the part of the sequence enclosed with ## and ##
        init_token = re.search(r"##(.*?)##", sequence).group(1)
        init_token = f"##{init_token}##"
        # expression token is the part of the sequence left without init token
        expression_token = sequence.replace(init_token, "").strip()
        assert expression_token[-1] == "$" and expression_token[0] == "$", (
            "<invalid-parse-expression>"
        )
        reactants, reaction, product = self._parse_expression_token(expression_token)
        return init_token, (reactants, reaction, product)

    def _validate_rxn_map(self, code, rxn_sequence):
        assert code in reaction_smarts_map, "<invalid-reaction-code>"
        assert rxn_sequence.strip() in reaction_smarts_map[code], (
            "<invalid-reaction-sequence>"
        )

    def _validate_contents(self, counts, expression_tokens):
        prod_sequence = expression_tokens[-1]
        rxn_sequence = expression_tokens[-2]
        cls_code, n_count, o_count, ring_count, poly_l = counts
        assert int(n_count) == _get_nonaromatic_nitrogen(prod_sequence), (
            "<invalid-content-N>"
        )
        assert int(o_count) == _get_nonaromatic_oxygen(prod_sequence), (
            "<invalid-content-O>"
        )
        assert int(ring_count) == _get_num_rings(prod_sequence), (
            "<invalid-content-ring>"
        )
        # adding 1 to the length as the syntax keeps a space between => and product
        assert int(poly_l) + 1 == len(prod_sequence), "<invalid-content-poly-length>"
        # validate the class code with specific reaction types
        self._validate_rxn_map(cls_code, rxn_sequence)

    def _validate_init_token(self, init_token, expression_tokens):
        values = init_token.strip("##").split(":")
        assert len(values) == 5, "<invalid-parse-init>"
        class_code, n_count, o_count, ring_count, poly_l = values
        assert class_code.isalpha(), "<invalid-parse-init-class-code>"
        # There is another layer of validation to validate if the code is mapped with the reaction SMARTS as valid or not

        assert class_code in ["U", "E", "A", "I", "Et"], "<invalid-class-code>"
        assert n_count.isdigit(), "<invalid-content-N>"
        assert o_count.isdigit(), "<invalid-content-O>"
        assert ring_count.isdigit(), "<invalid-content-ring>"
        assert poly_l.isdigit(), "<invalid-content-poly-l>"
        self._validate_contents(values, expression_tokens)

    def _validate_exp_tokens(self, expression_token):
        # Validate the expression token
        reactants, reaction, product = expression_token
        for reactant in reactants:
            try:
                Chem.MolFromSmiles(reactant.strip())
            except Exception as e:
                raise ValueError(f"<invalid-reactant-smiles> <{e}>")

        try:
            rxn = AllChem.ReactionFromSmarts(reaction.strip())
            rxn.Initialize()
            if rxn is None:
                raise ValueError("<invalid-reaction-smarts>")
            if rxn.GetNumReactantTemplates() != len(reactants):
                raise ValueError("<invalid-reactant-count>")
            try:
                prods = rxn.RunReactants(
                    [Chem.MolFromSmiles(reactant.strip()) for reactant in reactants]
                )
                if not prods:
                    raise ValueError("<invalid-product-count>")
                prods_smiles = [
                    list(Chem.MolToSmiles(prod, canonical=True) for prod in prod_s)
                    for prod_s in prods
                ]
                # flatten this prods_smiles list of lists to list of one dimensions
                prods_smiles = [item for sublist in prods_smiles for item in sublist]
                product_smiles = Chem.MolToSmiles(
                    Chem.MolFromSmiles(product.strip()), canonical=True
                )
                if product_smiles not in prods_smiles:
                    raise ValueError("<invalid-product-smiles>")
            except Exception as e:
                raise ValueError(f"<invalid-product-smiles> <{e}>")
        except Exception as e:
            raise ValueError(f"<invalid-reaction-smarts> <{e}>")

        return None

    def validate(self, sequence):
        """
        You have to modify this try and except block to return the error code to evaluate the failure causes and types to
        extract where the system actually fails.
        """
        if self.variation_code == 1:
            init_token, expression_tokens = self._parse(sequence)
            try:
                self._validate_init_token(init_token, expression_tokens)
                self._validate_exp_tokens(expression_tokens)
                return True, "<none>"
            except Exception as e:
                return False, str(e)
        else:
            print("[--] We currently do not validate for variation code 0")
            return False, "<unsupported-var-code>"


if __name__ == "__main__":
    # validator = StringValidatorBRICSBase()
    # validator.validate(
    #     "[5*]NCN[5*].[1*]Nc1ccc(NCCC[4*])cc1|[4*]-[*:1].[5*]-[*:2]>>[$([C&!D1&!$(C=*)]-&!@[#6]):1]-&!@[$([N&!D1&!$(N=*)&!$(N-[!#6&!#16&!#0&!#1])&!$([N&R]@[C&R]=O)]):2]|0.5|0.5|<1-2:0.375:0.375<1-1:0.375:0.375<2-2:0.375:0.375<3-4:0.375:0.375<3-3:0.375:0.375<4-4:0.125:0.125<1-3:0.125:0.125<1-4:0.125:0.125<2-3:0.125:0.125<2-4:0.125:0.125"
    # )

    # candidates for variation 1
    Homo_Candidate = "##E:0:4:1:27## ${Cc1cc(C(=O)O)c(O)c(C(=O)O)c1} => ([O&X2&H1&!$(OC=*):1].[C&X3:2](=O)[O&X2&H1])>>(*-[O&X2:1].[C&X3:2](=O)-*) => *Oc1c(C(*)=O)cc(C)cc1C(=O)O$"
    Co_Candidate = "##E:0:8:6:83## ${Cc1cc(CC(=O)O)c(C(=O)Cl)cc1C} + {O=C1c2ccccc2C(=O)c2cc(-c3ccc(O)cc3O)c(-c3cc(O)cc(O)c3)cc21} => ([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*) => *Oc1cc(O)cc(-c2cc3c(cc2-c2ccc(O)cc2OC(=O)Cc2cc(C)c(C)cc2C(*)=O)C(=O)c2ccccc2C3=O)c1$"

    # candidates for variation 0
    # Homo_Candidate = "##E:0:4:1:27## ${[[6*]C(=O)O] + [[16*]c1cc(C)cc(C(=O)O)c1O] -> [6*]-[*:1].[16*]-[*:2]>>[$([C&D3&!R](=O)-&!@[#0,#6,#7,#8]):1]-&!@[$([c&$(c(:c):c)]):2] -> [Cc1cc(C(=O)O)c(O)c(C(=O)O)c1]} => ([O&X2&H1&!$(OC=*):1].[C&X3:2](=O)[O&X2&H1])>>(*-[O&X2:1].[C&X3:2](=O)-*) => *Oc1c(C(*)=O)cc(C)cc1C(=O)O$"
    # Co_Candidate = "##E:0:8:6:83## ${[[6*]C(=O)Cl] + [[16*]c1cc(C)c(C)cc1CC(=O)O] -> [6*]-[*:1].[16*]-[*:2]>>[$([C&D3&!R](=O)-&!@[#0,#6,#7,#8]):1]-&!@[$([c&$(c(:c):c)]):2] -> [Cc1cc(CC(=O)O)c(C(=O)Cl)cc1C]} + {[[16*]c1ccc(O)cc1O] + [[16*]c1cc2c(cc1-c1cc(O)cc(O)c1)C(=O)c1ccccc1C2=O] -> [16*]-[*:1].[16*]-[*:2]>>[$([c&$(c(:c):c)]):1]-&!@[$([c&$(c(:c):c)]):2] -> [O=C1c2ccccc2C(=O)c2cc(-c3ccc(O)cc3O)c(-c3cc(O)cc(O)c3)cc21]} => ([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*) => *Oc1cc(O)cc(-c2cc3c(cc2-c2ccc(O)cc2OC(=O)Cc2cc(C)c(C)cc2C(*)=O)C(=O)c2ccccc2C3=O)c1$"

    validator = SeqValidator(variation_code=1)
    state, e_value = validator.validate(Homo_Candidate)
    print("state >>", state)
    print("e_value >>", e_value)
    state, e_value = validator.validate(Co_Candidate)
    print("state >>", state)
    print("e_value >>", e_value)
