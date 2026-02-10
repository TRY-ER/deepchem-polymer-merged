import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))  # Go up two levels to reach project root
sys.path.insert(0, project_root)

from src.discriminator.base import Discriminator
from deepchem import feat
from deepchem.data import NumpyDataset


class LogPDiscriminator(Discriminator):
    def __init__(self,
                 model_type: str,
                 model_dir: str):
        super().__init__(model_type, model_dir)
        self.model = self.get_model()
        self.featurizer = self.get_featurizer()

    def get_featurizer(self):
        if self.model_type == "dmpnn":
            return feat.DMPNNFeaturizer()
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def get_model(self):
        if self.model_type == "dmpnn":
            from deepchem.models import DMPNNModel
            model = DMPNNModel()
            model.restore(model_dir=self.model_dir)
            return model
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def predict(self, sequence: str | list[str]) -> float | list[float]:
        if isinstance(sequence, str):
            feature_vector = self.featurizer.featurize(sequence)
            dataset = NumpyDataset(feature_vector)
            predictions = self.model.predict(dataset)
            return predictions[0][0]

        elif isinstance(sequence, list):
            return_list = []
            for seq in sequence:
                feature_vector = self.featurizer.featurize(seq)
                dataset = NumpyDataset(feature_vector)
                predictions = self.model.predict(dataset)
                return_list.append(predictions[0][0])
            return return_list

        else:
            raise ValueError(
                "Input must be a valid string or a list of valid strings.")


if __name__ == "__main__":
    discriminator = LogPDiscriminator(model_type="dmpnn",
                                      model_dir="C:\\Users\\Kalki\\Documents\\code\\dfs\\models\\discriminator\\LogP Regressor\\DMPNN_model_dir")
    logp_value = discriminator.predict(["CCO", "CC(=O)"])
    print(f"Predicted logP value: {logp_value}")
    