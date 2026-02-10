from src.validator.validator import StringValidator
from rdkit import Chem

class Metric:
    """Base class for all metrics."""

    def __init__(self):
        pass

    def compute(self, *args, **kwargs):
        """Compute the metric.

        This method should be overridden by subclasses.

        Args:
            *args: Positional arguments for metric computation.
            **kwargs: Keyword arguments for metric computation.

        Returns:
            The computed metric value.
        """
        raise NotImplementedError("Subclasses should implement this method.")


class ValidityMetric(Metric):
    """Class for calculating validity metric."""

    def __init__(self):
        pass

    def compute(self, generated_samples: list[str]) -> float:
        """Compute the validity of generated samples.

        Args:
            generated_samples (list): List of generated samples (e.g., SMILES strings).

        Returns:
            float: Validity score as a percentage.
        """
        valid_count = sum(1 for sample in generated_samples if self.is_valid(sample))
        return valid_count / len(generated_samples) * 100

    def get_valid_set(self, generated_samples):
        return set(filter(self.is_valid, generated_samples))

    @staticmethod
    def is_valid(sample: str) -> bool:
        """Check if a sample is valid.

        Args:
            sample: A single generated sample.

        Returns:
            bool: True if the sample is valid, False otherwise.
        """
        validator = StringValidator()
        return validator.validate(sample)


class UniquenessMetric(Metric):
    """Class for calculating uniqueness metric."""

    def __init__(self):
        pass

    def cannonicalize(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        return Chem.MolToSmiles(mol, canonical=True)

    def process_single(self, candidate: str) -> str:
        smiles_part, _, weight_part = StringValidator.parse_parts(candidate)
        smiles_part = smiles_part.replace(" ", "")
        returnable = []
        for smiles in smiles_part.split("."):
            can = self.cannonicalize(smiles)
            returnable.append(can)
        return ".".join(sorted(returnable)) + "|" + weight_part

    def get_uniqueness_set(self, generated_samples):
        return set(map(self.process_single, generated_samples))

    def compute(self, generated_samples):
        """
        Compute the uniqueness of generated samples.

        Args:
            generated_samples (list): List of generated samples (e.g., SMILES strings).

        Returns:
            float: Uniqueness score as a percentage.
        """
        comp_strings = self.get_uniqueness_set(generated_samples)
        return len(comp_strings) / len(generated_samples) * 100


class NoveltyMetric(Metric):
    """Class for calculating novelty metric."""

    def __init__(self, training_data: list[str]):
        self.training_data = training_data

    def compute(self, generated_samples: list[str]) -> float:
        """Compute the novelty of generated samples.

        Args:
            generated_samples (list): List of generated samples (e.g., SMILES strings).

        Returns:
            float: Novelty score as a percentage.
        """
        training_formatted = UniquenessMetric().get_uniqueness_set(
            self.training_data)
        generated_formatted = UniquenessMetric().get_uniqueness_set(
            generated_samples)

        novel_count = sum(1 for sample in generated_formatted if sample not in training_formatted) 
        return novel_count / len(
            generated_samples) * 100

class ChainFormLikelyHood(Metric):
    """Class for calculating Chain Form LikelyHood metric."""

    def __init__(self):
        pass

    def compute(self, generated_samples):
        """Compute the Chain Form LikelyHood of generated samples.

        Args:
            generated_samples (list): List of generated samples (e.g., SMILES strings).

        Returns:
            float: Chain Form LikelyHood score as a percentage.
        """
        # Placeholder implementation
        return 0.0  # Replace with actual computation logic

class SubStructureValidityMetrics(Metric):
    """Class for calculating SubStructure Validity metric."""

    def __init__(self, substructures):
        self.substructures = substructures

    def compute(self, generated_samples):
        """Compute the SubStructure Validity of generated samples.

        Args:
            generated_samples (list): List of generated samples (e.g., SMILES strings).

        Returns:
            float: SubStructure Validity score as a percentage.
        """
        valid_count = sum(1 for sample in generated_samples if self.contains_substructure(sample))
        return valid_count / len(generated_samples) * 100 if generated_samples else 0.0

    def contains_substructure(self, sample):
        """Check if a sample contains any of the specified substructures.

        Args:
            sample: A single generated sample.

        Returns:
            bool: True if the sample contains any substructure, False otherwise.
        """
        return any(sub in sample for sub in self.substructures)  # Example check

class MasterMetric(Metric):
    """Class for calculating a master metric combining multiple metrics."""

    def __init__(self, generated_samples: list[str]):
        self.generated_samples = generated_samples
        self.total_length = len(generated_samples)
        self.validity_score = None
        self.uniqueness_score = None
        self.novelty_score = None
        self.chain_form_likelihood_score = None
        self.substructure_validity_score = None


    def compute(self):
        """Compute all metrics and return a summary.

        Returns:
            dict: A dictionary containing all computed metric scores.
        """
        self.validity_score = ValidityMetric().compute(self.generated_samples)
        self.uniqueness_score = UniquenessMetric().compute(self.generated_samples)
        self.novelty_score = NoveltyMetric(training_data=[]).compute(self.generated_samples)  # Replace with actual training data
        self.chain_form_likelihood_score = ChainFormLikelyHood().compute(self.generated_samples)
        self.substructure_validity_score = SubStructureValidityMetrics(substructures=[]).compute(self.generated_samples)  # Replace with actual substructures

        return {
            "validity": self.validity_score,
            "uniqueness": self.uniqueness_score,
            "novelty": self.novelty_score,
            "chain_form_likelihood": self.chain_form_likelihood_score,
            "substructure_validity": self.substructure_validity_score
        }
