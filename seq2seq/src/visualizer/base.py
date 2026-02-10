# load src into python path
import sys
import os
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))



from src.validator.validator import StringValidator
from rdkit import Chem
import copy
import json
from rdkit.Chem import Draw
import matplotlib
import matplotlib.pyplot as plt
from collections import defaultdict
import cairosvg
from io import BytesIO

# Optional progress bar for batch operations
try:
    from tqdm import tqdm
except Exception:
    tqdm = None

matplotlib.use('Agg')  # Set non-GUI backend before importing pyplot


class CustomVisualizer:
    """Custom Class for Visualizing Polymer Structures with Additional Annotations and Filter Weights."""

    def __init__(self,
                 canvas_size: tuple[int, int] = (1400, 500),
                 figure_size: tuple[int, int] = (10, 6),
                 output_dir: str = "./visualizations",
                 logP_color: dict[str, tuple] | None = None):
        self.canvas_size = canvas_size
        self.figure_size = figure_size
        self.output_dir = output_dir
        self.setup_output_dir()
        if not logP_color:
            self.logP_color = {
                "philic": (0.0, 1.0, 1.0, 1),  # Blue (RGB Alpha )
                "phobic": (1.0, 1.0, 0.0, 1),  # Yellow (RGB Alpha )
            }

    def setup_output_dir(self):
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def find_bond_indicies(self, smarts_str: str):
        # print("smarts_str >>", smarts_str)
        reactant_cont = smarts_str.split(">>")[0]
        parts = reactant_cont.split(".")
        assert len(parts) == 2, "Expected 2 parts in the reactant container"
        wildcard1 = parts[0].split("-")[0]
        wildcard2 = parts[1].split("-")[0]
        return wildcard1, wildcard2

    def form_extrabond(self, rwmol, smarts_part):
        part1, part2 = [], []
        for atom in rwmol.GetAtoms():
            smarts = atom.GetSmarts()
            if "*" in smarts:
                if len(part1) == 2:
                    part2.append({atom.GetIdx(): smarts})
                else:
                    part1.append({atom.GetIdx(): smarts})

        wildcard1, wildcard2 = self.find_bond_indicies(smarts_part)
        comp = []
        if any(wildcard1 in part.values()
               for part in part1) and any(wildcard1 in part.values()
                                          for part in part2):
            if any(wildcard2 in part.values() for part in part1):
                comp = [wildcard2, wildcard1]
            else:
                comp = [wildcard1, wildcard2]
        elif any(wildcard2 in part.values()
                 for part in part1) and any(wildcard2 in part.values()
                                            for part in part2):
            if any(wildcard1 in part.values() for part in part1):
                comp = [wildcard1, wildcard2]
            else:
                comp = [wildcard2, wildcard1]
        elif any(wildcard1 in part.values() for part in part1):
            comp = [wildcard1, wildcard2]
        elif any(wildcard2 in part.values() for part in part1):
            comp = [wildcard2, wildcard1]
        else:
            comp = [wildcard1, wildcard2]
        # print("wildcard1 >>", wildcard1)
        # print("wildcard2 >>", wildcard2)
        # print("part1 >>", part1)
        # print("part2 >>", part2)
        bond_index1, bond_index2 = -1, -1
        for part in part1:
            for idx, smarts in part.items():
                # for wildcard in comp:
                if comp[0] in smarts:
                    bond_index1 = idx
                    # comp.remove(wildcard)
        for part in part2:
            for idx, smarts in part.items():
                # for wildcard in comp:
                if comp[1] in smarts:
                    bond_index2 = idx
        # print("bond_index1 >>", bond_index1)
        # print("bond_index2 >>", bond_index2)
        copy_mol = copy.deepcopy(rwmol)
        copy_mol.AddBond(bond_index1, bond_index2, Chem.BondType.SINGLE)
        for bond in copy_mol.GetBonds():
            if bond.GetBeginAtomIdx() in [
                    bond_index1, bond_index2
            ] and bond.GetEndAtomIdx() in [bond_index1, bond_index2]:
                bond.SetProp("reaction_SMARTS", smarts_part)
        return copy_mol, (wildcard1, wildcard2), smarts_part

    def parse_weight(self, weight_part: str):
        weight_split = weight_part.split("|")
        prop_w1 = weight_split[1]
        prop_w2 = weight_split[2]
        rules = weight_split[3:]
        rules = rules[0].split("<")[1:]
        rule_weights = []
        for r in rules:
            atoms, weight1, weight2 = r.split(":")
            atoms = list(map(int, atoms.split("-")))
            # print("atoms >>", atoms)
            # print("weight1 >>", weight1)
            # print("weight2 >>", weight2)
            assert weight1 == weight2, "Expected weight1 and weight2 to be the same for current implementation"
            rule_weights.append((atoms, float(weight1)))
        return prop_w1, prop_w2, rule_weights

    def form_atom_weights(self, mol, weight_part: str):
        atom_w1, atom_w2, rule_weights = self.parse_weight(weight_part)
        atom_map = {}
        prop_map = {}
        counter = 1
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                atom_map[counter] = atom.GetSmarts()
                counter += 1
        for atom in atom_map.keys():
            content = {}
            for rule in rule_weights:
                atoms, weight = rule
                if atom in atoms:
                    try:
                        alt_ind = [x for x in atoms if x != atom][0]
                        content[f"{atom_map[alt_ind]}"] = weight
                    except Exception as e:
                        # <debug print>
                        # print("Error processing rule:", rule,
                        #   "for atom:", atom, "Error:", e)
                        ...
            prop_map[atom] = content
        l_map = ["(a)", "(b)", "(c)", "(d)"]
        counter = 0
        prop_export = {}
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "*":
                props = prop_map.get(counter + 1, {})
                prop_export[l_map[counter]] = props
                # <debug print>
                # print(f"props found for {counter + 1} >>", props)
                atom.SetProp("atomNote", l_map[counter])
                counter += 1
        return mol, prop_export

    def draw_plot(self, new_mol, wildcard1, wildcard2, smarts_part,
                  atomHighlights, bondHighlights, atomRads,
                  prop_export):
        # Draw molecule without notes
        # img = Draw.MolToImage(new_mol, size=self.canvas_size)
        # d2d = Draw.MolDraw2DSVG(self.canvas_size[0], self.canvas_size[1])
        d2d = Draw.MolDraw2DCairo(self.canvas_size[0], self.canvas_size[1])
        dopts = d2d.drawOptions()
        # dopts.useBWAtomPalette()  # Commented out - causes segfault
        # I think this looks better if we ensure that the atom highlights are always circles:
        dopts.atomHighlightsAreCircles = True
        dopts.baseFontSize = 1
        dopts.annotationFontScale = 0.7
        d2d.DrawMoleculeWithHighlights(new_mol, "", dict(atomHighlights),
                                       dict(bondHighlights), atomRads, {})

        d2d.FinishDrawing()
        # png_data = BytesIO()
        # # Convert SVG to PNG
        # cairosvg.svg2png(bytestring=d2d.GetDrawingText().encode('utf-8'), write_to=png_data)
        # png_data.seek(0)

        # # Create matplotlib figure
        # assert png_data is not None, "Failed to convert SVG to PNG."
        png_data = BytesIO(d2d.GetDrawingText())
        img = plt.imread(png_data)
        fig, ax = plt.subplots(figsize=self.figure_size)
        ax.imshow(img)
        ax.axis('off')

        # Add text box for long props (multiline supported)
        props_text = f"Reaction SMARTS Between {wildcard1} and {wildcard2}: \n\n{smarts_part}\n\n"
        props_text = props_text.replace('$', '\\$')

        ax.text(0.5,
                -0.1,
                props_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='center',
                bbox=dict(facecolor='white',
                          edgecolor='black',
                          boxstyle='round,pad=0.5'))

        # Add legend for hydrophobic/hydrophilic sections
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.logP_color["phobic"],
                  label='Hydrophobic (logP ≥ 0)'),
            Patch(facecolor=self.logP_color["philic"],
                  label='Hydrophilic (logP < 0)')
        ]
        ax.legend(handles=legend_elements,
                  loc='upper right',
                  bbox_to_anchor=(1.0, 1.0))

        # Add prop_export legend in the upper left
        prop_legend = ""
        for i, (key, value) in enumerate(prop_export.items()):
            if i != len(prop_export) - 1:
                prop_legend += f"{key} → {json.dumps(value, separators=(',', ':'))}\n"
            else:
                prop_legend += f"{key} → {json.dumps(value, separators=(',', ':'))}"
        
        if prop_legend:
            ax.text(0, 1.0, prop_legend,
                   transform=ax.transAxes,
                   fontsize=8,
                   verticalalignment='top',
                   horizontalalignment='left',
                   family='monospace',
                   bbox=dict(facecolor='white',
                            edgecolor='black',
                            alpha=0.8,
                            boxstyle='round,pad=0.5'))

        return fig

    def show_image(self, new_mol, wildcard1, wildcard2, smarts_part,
                   atomHighlights, bondHighlights, atomRads, prop_export):
        fig = self.draw_plot(new_mol, wildcard1, wildcard2, smarts_part,
                             atomHighlights, bondHighlights, atomRads, prop_export)
        fig.show()

    def get_filter_colors(self, logP_1: float, logP_2: float):
        if logP_1 >= 0:
            color1 = self.logP_color["phobic"]
        else:
            color1 = self.logP_color["philic"]
        if logP_2 >= 0:
            color2 = self.logP_color["phobic"]
        else:
            color2 = self.logP_color["philic"]
        return color1, color2

    def get_result_mol(self,
                       data: str,
                       logP_filter: dict[str, float] | None = None):
        validator = StringValidator()
        if validator.validate(data):
            smiles_part, smarts_part, weights_part = StringValidator.parse_parts(
                data)  # Example usage of parsing
            # print("smiles part >>", smiles_part)
            rw_mol = Chem.RWMol(Chem.MolFromSmiles(smiles_part))
            mol, (wildcard1, wildcard2), smarts_part = self.form_extrabond(
                rw_mol, smarts_part)
            new_mol, prop_export = self.form_atom_weights(mol, weights_part)
            atomHighlights = defaultdict(list)
            bondHighlights = defaultdict(list)
            atomRads = {}
            if logP_filter is not None:
                assert "logP_1" in logP_filter and "logP_2" in logP_filter, "Both logP_1 and logP_2 must be provided in the filter."
                logP_1 = logP_filter["logP_1"]
                logP_2 = logP_filter["logP_2"]
                assert len(
                    smiles_part.split(".")
                ) == 2, "Expected 2 parts in the smiles part for logP filtering."
                smiles_1, smiles_2 = smiles_part.split(".")
                matches1 = new_mol.GetSubstructMatches(
                    Chem.MolFromSmiles(smiles_1))
                matches2 = new_mol.GetSubstructMatches(
                    Chem.MolFromSmiles(smiles_2))
                color1, color2 = self.get_filter_colors(logP_1, logP_2)

                rad = 0.3

                for atom in new_mol.GetAtoms():
                    atom_idx = atom.GetIdx()
                    if atom_idx in matches1[0]:
                        atomHighlights[atom_idx].append(color1)
                        atomRads[atom_idx] = rad
                    if atom_idx in matches2[0]:
                        atomHighlights[atom_idx].append(color2)
                        atomRads[atom_idx] = rad

                for bond in new_mol.GetBonds():
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    if (begin_idx in matches1[0]) and (end_idx in matches1[0]):
                        bondHighlights[bond.GetIdx()].append(color1)
                    if (begin_idx in matches2[0]) and (end_idx in matches2[0]):
                        bondHighlights[bond.GetIdx()].append(color2)

            return new_mol, (
                wildcard1, wildcard2
            ), smarts_part, atomHighlights, bondHighlights, atomRads, prop_export
        else:
            raise ValueError("Invalid data format for visualization.")

    def visualize(self,
                  data: str,
                  logP_filter: dict[str, float] | None = None):
        if isinstance(data, str):
            (new_mol, (wildcard1,
                       wildcard2), smarts_part, atomHighlights, bondHighlights,
             atomRads,
             prop_export) = self.get_result_mol(data, logP_filter)
            self.show_image(new_mol, wildcard1, wildcard2, smarts_part,
                            atomHighlights, bondHighlights, atomRads, prop_export)
            plt.close()
        else:
            raise ValueError("Invalid data format for visualization.")

    def save_image(self,
                   data: str,
                   filename: str,
                   logP_filter: dict[str, float] | None = None):
        if isinstance(data, str):
            (new_mol, (wildcard1,
                       wildcard2), smarts_part, atomHighlights, bondHighlights,
             atomRads, prop_export) = self.get_result_mol(data, logP_filter)
            fig = self.draw_plot(new_mol, wildcard1, wildcard2, smarts_part,
                                 atomHighlights, bondHighlights, atomRads,
                                 prop_export)
            fig_path = os.path.join(self.output_dir, filename)
            fig.savefig(fig_path, bbox_inches='tight')
            plt.close()
        else:
            raise ValueError("Invalid data format for visualization.")

    def save_image_batch(self,
                         data_list: list[str],
                         output_file_intro: str = "output_image",
                         logP_filter: list[dict[str, float]] | None = None):
        """Save a batch of images; shows a progress bar when tqdm is available.

        Args:
            data_list: list of polymer data strings (SMILES+SMARTS+weights format)
            output_file_intro: prefix for output filenames
            logP_filter: optional list of logP filter dicts, one per data item
        """
        if logP_filter is not None:
            assert len(data_list) == len(
                logP_filter), "Length of data_list and logP_filter must match."

        # Choose iterator with or without tqdm
        if tqdm:
            iterator = enumerate(tqdm(data_list,
                                      desc="Generating visualizations",
                                      unit="img"),
                                 start=1)
        else:
            iterator = enumerate(data_list, start=1)

        for idx, data in iterator:
            logP_dict = None
            if logP_filter is not None:
                logP_dict = logP_filter[idx - 1]
            filename = f"{output_file_intro}_{idx}.png"
            self.save_image(data, filename, logP_dict)
            plt.close()


if __name__ == "__main__":
    visualizer = CustomVisualizer()
    # test_str = "[1*]C(=O)C[4*].[4*]CC(O)COC(=O)CCCCCN[5*]|[4*]-[*:1].[5*]-[*:2]>>[$([C&!D1&!$(C=*)]-&!@[#6]):1]-&!@[$([N&!D1&!$(N=*)&!$(N-[!#6&!#16&!#0&!#1])&!$([N&R]@[C&R]=O)]):2]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5"
    # visualizer.visualize(test_str,
    #                      logP_filter={
    #                          "logP_1": -1.2,
    #                          "logP_2": 2.3
    #                      }
    #                      )
    # visualizer.save_image(test_str, logP_filter= {
    #     "logP_1": -1.2,
    #     "logP_2": 2.3
    # }, filename="output_image.png")

    test_batch = [
        "[1*]C(=O)C[4*].[4*]CC(O)COC(=O)CCCCCN[5*]|[4*]-[*:1].[5*]-[*:2]>>[$([C&!D1&!$(C=*)]-&!@[#6]):1]-&!@[$([N&!D1&!$(N=*)&!$(N-[!#6&!#16&!#0&!#1])&!$([N&R]@[C&R]=O)]):2]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5",
        "[1*]C([1*])=O.[3*]OCCCNCN[5*]|[1*]-[*:1].[5*]-[*:2]>>[$([C&D3]([#0,#6,#7,#8])=O):1]-&!@[$([N&!D1&!$(N=*)&!$(N-[!#6&!#16&!#0&!#1])&!$([N&R]@[C&R]=O)]):2]|0.5|0.5|<1-3:0.5:0.5<1-4:0.5:0.5<2-3:0.5:0.5<2-4:0.5:0.5",
    ]

    logP_filters = [
        {
            "logP_1": -1.2,
            "logP_2": 2.3
        },
        {
            "logP_1": 0.5,
            "logP_2": -0.8
        }
    ]


    visualizer.save_image_batch(test_batch,
                                output_file_intro="batch_output_image",
                                logP_filter=logP_filters)
