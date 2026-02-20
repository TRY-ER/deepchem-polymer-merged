import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach project root
sys.path.insert(0, project_root)

from src.discriminator.logP_discriminator import LogPDiscriminator
import re
import numpy as np


class LogPFilter:
    def __init__(self,
                 model_type: str,
                 model_dir: str,
                 logp_threshold: float):
        self.discriminator = LogPDiscriminator(model_type, model_dir)
        self.logp_threshold = logp_threshold

    def get_smiles_split(self, sequence: str) -> list[str]:
        sequence = sequence.replace(" ", "")
        smiles_part = sequence.split("|")[0]
        # Remove patterns like [5*] using regex
        smiles_part = re.sub(r'\[\d+\*\]', '', smiles_part)
        smiles_part = smiles_part.replace("()", "")
        parts = smiles_part.split('.')
        assert len(
            parts) == 2, "The SMILES does not split into two parts as expected."
        return parts

    def find_diff(self, val1: float | np.float32, val2: float | np.float32) -> float | np.float32:
        if val1 > val2:
            return val1 - val2
        else:
            return val2 - val1

    def validate_diff(self, val1: float | np.float32, val2: float | np.float32) -> int:
        if (val1 < 0 and val2 > 0) or (val1 > 0 and val2 < 0):
            return 1
        else:
            return 0

    def preprocess_sequence(self, sequence: str) -> str:
        sequence = sequence.replace(" ", "")
        sequence = sequence.replace('"', '')
        sequence = sequence.replace("'", '')
        return sequence

    def get_analysis(self, parts: list[str]):
        logP_1 = self.discriminator.predict(parts[0])
        logP_2 = self.discriminator.predict(parts[1])
        assert type(logP_1) in (float, np.float32), "Predicted logP1 value is not a float."
        assert type(logP_2) in (float, np.float32), "Predicted logP2 value is not a float."
        logP_delta = self.find_diff(logP_1, logP_2)
        validate = self.validate_diff(logP_1, logP_2)
        if validate == 1 and logP_delta > self.logp_threshold:
            return True, parts, (logP_1, logP_2, logP_delta)
        else:
            return False, parts, (logP_1, logP_2, logP_delta)

    def filter(self, sequence: str | list[str]):
        if isinstance(sequence, str):
            parts = self.get_smiles_split(self.preprocess_sequence(sequence))
            return self.get_analysis(parts)

        elif isinstance(sequence, list):
            proccessed_results = map(self.preprocess_sequence, sequence)
            parts_list = list(map(self.get_smiles_split, proccessed_results))
            return [self.get_analysis(parts) for parts in parts_list]


if __name__ == "__main__":
    # change the model dir before use
    filter = LogPFilter(model_type="dmpnn",
                        model_dir="C:\\Users\\Kalki\\Documents\\code\\dfs\\models\\discriminator\\LogP Regressor\\DMPNN_model_dir",
                        logp_threshold=0.5)
    return_vals = filter.filter([
        "[ 1 * ] C ( = O ) C ( [ 4 * ] ) C. [ 3 * ] OC ( = O ) CCCCCCCCCCC ( = O ) OC ( = O ) CCCCC [ 4 * ] | [ 1 * ] - [ * : 1 ]. [ 3 * ] - [ * : 2 ] > > [ $ ( [ C & D3 ] ( [ # 0, # 6, # 7, # 8 ] ) = O ) : 1 ] - &! @ [ $ ( [ O & D2 ] - &! @ [ # 0, # 6, # 1 ] ) : 2 ] | 0. 5 | 0. 5 | < 1 - 2 : 0. 375 : 0. 375 < 1 - 1 : 0. 375 : 0. 375 < 2 - 2 : 0. 375 : 0. 375 < 3 - 4 : 0. 375 : 0. 375 < 3 - 3 : 0. 375 : 0. 375 < 4 - 4 : 0. 125 : 0. 125 < 1 - 3 : 0. 125 : 0. 125 < 1 - 4 : 0. 125 : 0. 125 < 2 - 3 : 0. 125 : 0. 125 < 2 - 4 : 0. 125 : 0. 125",
        "[ 4 * ] CC ( O ) C [ 4 * ]. [ 1 * ] C ( = O ) C ( O ) C ( = O ) C ( O ) C ( = O ) C ( O ) C ( = O ) C ( [ 3 * ] ) O | [ 3 * ] - [ * : 1 ]. [ 4 * ] - [ * : 2 ] > > [ $ ( [ O & D2 ] - &! @ [ # 0, # 6, # 1 ] ) : 1 ] - &! @ [ $ ( [ C &! D1 &! $ ( C = * ) ] - &! @ [ # 6 ] ) : 2 ] | 0. 5 | 0. 5 | < 1 - 2 : 0. 375 : 0. 375 < 1 - 1 : 0. 375 : 0. 375 < 2 - 2 : 0. 375 : 0. 375 < 3 - 4 : 0. 375 : 0. 375 < 3 - 3 : 0. 375 : 0. 375 < 4 - 4 : 0. 125 : 0. 125 < 1 - 3 : 0. 125 : 0. 125 < 1 - 4 : 0. 125 : 0. 125 < 2 - 3 : 0. 125 : 0. 125 < 2 - 4 : 0. 125 : 0. 125"
    ]
    )

    print(f"Filter return values: {return_vals}")
