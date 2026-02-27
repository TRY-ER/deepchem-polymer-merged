import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)


import matplotlib.pyplot as plt

from src.validator.validator import SeqValidator

METRICS_ERROR_MAP = {
    "<invalid-parse-expression>",
    "<invalid-reaction-code>",
    "<invalid-reaction-sequence>",
    "<invalid-content-N>",
    "<invalid-content-O>",
    "<invalid-content-ring>",
    "<invalid-content-poly-length>",
    "<invalid-parse-init>",
    "<invalid-parse-init-class-code>",
    "<invalid-class-code>",
    "<invalid-reactant-smiles>",
    "<invalid-reaction-smarts>",
    "<invalid-reactant-count>",
    "<invalid-product-count>",
    "<invalid-product-smiles-match>",
    "<invalid-generated-product-smiles>",
    "<invalid-reaction-product-smiles>",
    "<unsupported-var-code>",
}


class Seq2SeqValidityMetrics:
    def __init__(self, df, col_name):
        self.df = df
        self.col_name = col_name
        self.metrics = {metric_id: 0 for metric_id in METRICS_ERROR_MAP}
        self.eval = None

    def _validate(self, row):
        validator = SeqValidator()
        errors = validator.validate(
            row[self.col_name]
            .replace(" ", "")
            .replace("=>", " => ")  # this needs to be replace later to post process
        )

        for error_code in METRICS_ERROR_MAP:
            row[error_code] = 0
        for error in set(errors):
            if error in METRICS_ERROR_MAP:
                row[error] = 1
                self.add_to_metric(error, 1)
        return errors

    def add_to_metric(self, name, value):
        if name not in self.metrics:
            raise ValueError(f"Invalid metric name: {name}")
        self.metrics[name] = self.metrics[name] + value

    def evaluate(self):
        self.df.apply(self._validate, axis=1)
        self.eval = {
            metric_id: (metric_value / self.df.shape[0])
            for metric_id, metric_value in self.metrics.items()
        }
        return self.eval

    def compute(self):
        self.evaluate()
        validation_score = -1
        if self.eval:
            validation_score = 1 - (
                sum([self.eval[metric_id] for metric_id in self.eval]) / len(self.eval)
            )

        return validation_score, self.eval

    def visualize(self, save_path=None):
        """
        Visualizes the error distribution from METRICS_ERROR_MAP, excluding <unsupported-var-code>.
        """
        if self.eval is None:
            self.evaluate()

        # Filter out <unsupported-var-code> as requested
        display_metrics = {
            k: v for k, v in self.eval.items() if k != "<unsupported-var-code>"
        }

        # Sort metrics by value for better readability
        sorted_metrics = sorted(
            display_metrics.items(), key=lambda x: x[1], reverse=True
        )
        names = [item[0] for item in sorted_metrics]
        values = [item[1] for item in sorted_metrics]

        plt.figure(figsize=(12, 8))
        plt.barh(names, values, color="skyblue")
        plt.gca().invert_yaxis()  # Put the highest error rate at the top
        plt.xlabel("Error Rate (Frequency)")
        plt.title("Seq2Seq Validity Metrics Error Distribution")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


if __name__ == "__main__":
    import pandas as pd

    # Example usage
    df = pd.DataFrame(
        {
            "sequence": [
                "##E:0:4:1:27## ${Cc1cc(C(=O)O)c(O)c(C(=O)O)c1} => ([O&X2&H1&!$(OC=*):1].[C&X3:2](=O)[O&X2&H1])>>(*-[O&X2:1].[C&X3:2](=O)-*) => *Oc1c(C(*)=O)cc(C)cc1C(=O)O$",
                "##E:0:8:6:83## ${Cc1cc(CC(=O)X)c(C(=O)Cl)cc1C} + {O=C1c2ccccc2C(=O)c2cc(-c3ccc(O)cc3O)c(-c3cc(O)cc(O)c3)cc21} => ([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*) => *Oc1cc(O)cc(-c2cc3c(cc2-c2ccc(O)cc2OC(=O)Cc2cc(C)c(C)cc2C(*)=O)C(=O)c2ccccc2C3=O)c1$",
                "##E:0:8:6:89## ${Cc1cc(CC(=O)X)c(C(=O)Cl)cc1C} + {O=C1c2ccccc2C(=O)c2cc(-c3ccc(O)cc3O)c(-c3cc(O)cc(O)c3)cc21} => ([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*) => *Oc1cc(O)cc(-c2cc3c(cc2-c2ccc(O)cc2OC(=O)Cc2cc(C)c(C)cc2C(*)=O)C(=O)c2ccccc2C3=O)c1$",
            ]
        }
    )
    metrics = Seq2SeqValidityMetrics(df, "sequence")
    score, _ = metrics.compute()
    metrics.visualize(save_path="./metrics_plot.png")
    print("score >>", score)
