import collections
import re
from turtle import resetscreen
from typing import Dict, List, Optional

import defaults


class GromacsTopUpdater:
    """
    Class to automatically update [ molecules ] section in GROMACS .top files
    based on residue counts from .gro coordinate files.

    Usage:
    updater = GromacsTopUpdater('topol.top')
    mol_counts = updater.count_molecules_from_gro('system.gro')
    updater.update_molecules(mol_counts)
    updater.write('topol_updated.top')
    """

    def __init__(self, top_file: str):
        self.top_file = top_file
        self.lines: List[str] = []
        self.molecules_section_start: Optional[int] = None
        self.molecules_section_end: Optional[int] = None
        self._load_topology()

    def _load_topology(self):
        """Load and parse topology file."""
        with open(self.top_file, "r") as f:
            self.lines = f.readlines()
        self._find_molecules_section()

    def _find_molecules_section(self):
        """Locate [ molecules ] section indices."""
        for i, line in enumerate(self.lines):
            if re.match(r"^\s*\[ molecules \]\s*$", line):
                self.molecules_section_start = i
                # Find end: next empty line or section after molecule entries
                for j in range(i + 1, len(self.lines)):
                    if re.match(r"^\s*(;.*)?\s*$", self.lines[j]) and j > i + 1:
                        self.molecules_section_end = j
                        break
                if self.molecules_section_end is None:
                    self.molecules_section_end = len(self.lines)
                return
        raise ValueError("No [ molecules ] section found in topology!")

    def count_molecules_from_top(self):
        """ """
        mol_section = self.lines[
            self.molecules_section_start + 1 : self.molecules_section_end
        ]
        mol_counts = {}
        for line in mol_section:
            if line.strip():
                resname, count = line.split()
                mol_counts[resname] = int(count)
        return mol_counts

    def count_solvent_from_gro(self, gro_file, solvent_resname):
        """ """
        with open(gro_file, "r") as f:
            lines = f.readlines()
        solvent_count = 0
        for line in lines[2:]:
            resname = line.split()[0]
            # remove the initial integers from the resname using re
            resname = re.sub(r"^\d+", "", resname)
            if resname == solvent_resname:
                solvent_count += 1
        return solvent_count

    def update_molecules(
        self,
        gro_file_path,
        mod_resname,
        mod_count,
        solvent_resname,
    ):
        """Replace [ molecules ] section with new counts."""
        if self.molecules_section_start is None:
            raise ValueError("Molecules section not found!")

        # Generate new [ molecules ] section
        new_section = ["[ molecules ]\n"]
        solvent_count = self.count_solvent_from_gro(gro_file_path, solvent_resname)
        mol_count = self.count_molecules_from_top()
        for resname, count in mol_count.items():
            val_count = count
            if resname == mod_resname:
                val_count = mod_count
            elif resname == solvent_resname:
                val_count = solvent_count
            new_section.append(f"{resname} {val_count}\n")
        new_section.append("\n")  # Blank line separator

        # Replace in lines
        end_idx = (
            self.molecules_section_end
            if self.molecules_section_end
            else len(self.lines)
        )
        self.lines = (
            self.lines[: self.molecules_section_start]
            + new_section
            + self.lines[end_idx:]
        )

    def write(self, output_file: Optional[str] = None) -> str:
        """Write updated topology. Returns output path."""
        if output_file is None:
            output_file = self.top_file.replace(".top", "_updated.top")

        with open(output_file, "w") as f:
            f.writelines(self.lines)
        print(f"Updated topology written to {output_file}")
        return output_file


class MDPWritter:
    def __init__(self, write_type: str, output_file: str):
        if write_type == "prod":
            self.params = defaults.md_prod_default_params
        elif write_type == "npt_equil":
            self.params = defaults.npt_default_params
        elif write_type == "em_min":
            self.params = defaults.em_default_params
        else:
            raise ValueError(f"Invalid type: {write_type}")
        self.output_file = output_file
        self.content = ""
        self.extra_params = {
            "comment_extra": "Extra parameters for the MDP file from the user",
            "comment_extra_line": "===============================================",
        }

    def override_params(self, override_params: Dict[str, str]):
        for key, value in override_params.items():
            if key in self.params:
                self.params[key] = value
            else:
                self.extra_params[key] = value

    def stringfy(self, key, value):
        if key.startswith("comment"):
            return f"\n; {value}\n"
        else:
            return f"{key}\t=\t{value}\n"

    def compose(self):
        for key, value in self.params.items():
            self.content += self.stringfy(key, value)
        for key, value in self.extra_params.items():
            self.content += self.stringfy(key, value)

    def write(self, params: Dict[str, str] | None = None, logger=None) -> str:
        if params:
            self.override_params(params)
        self.compose()
        with open(self.output_file, "w") as f:
            f.write(self.content)
        return self.output_file


# Example usage (implement this):
if __name__ == "__main__":
    updater = GromacsTopUpdater("./outputs/test/test_mol.top")
    updater.update_molecules(
        "./outputs/test/test_system.gro", "TEST_CHAIN_MOL", 10, "W"
    )
    updater.write()
