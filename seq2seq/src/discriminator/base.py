from src.discriminator.dconfig import ALLOWED_MODEL_TYPES


class Discriminator:
    def __init__(self,
                 model_type: str,
                 model_dir: str):
        if model_type not in ALLOWED_MODEL_TYPES:
            raise ValueError(f"Model type '{model_type}' is not allowed. The model type must be one of {ALLOWED_MODEL_TYPES}.")
        self.model_type = model_type
        self.model_dir = model_dir

    def get_featurizer(self):
        raise NotImplementedError("Subclasses must implement get_featurizer method.")

    def get_model(self):
        raise NotImplementedError("Subclasses must implement get_model method.")

    def predict(self, sequence: str | list[str] ) -> float | list[float]:
        raise NotImplementedError("Subclasses must implement predict method.")