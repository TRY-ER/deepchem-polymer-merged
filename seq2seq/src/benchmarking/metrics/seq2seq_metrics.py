import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)


import matplotlib.pyplot as plt

from src.validator.validator import SeqValidator

METRICS_ERROR_MAP = {
    # syntactic validation
    "synthetic": [
        "<invalid-init-sequence>",
        "<invalid-gen-sequence>",
        "<invalid-reactant-smiles>",
        "<invalid-reaction-smarts>",
    ],
    # functional validation
    "functional": [
        "<invalid-reaction-setup>",
        "<invalid-product-count>",
        "<invalid-reaction-product-smiles>",
        "<invalid-generated-product-smiles>",
        "<invalid-product-psmiles-match>",
        "<invalid-product-psmiles-wildcard>",
    ],
    # alignment validation
    "alignment": [
        "<invalid-content-N>",
        "<invalid-content-O>",
        "<invalid-content-ring>",
        "<invalid-content-poly-length>",
        "<invalid-reaction-code-sequence-match>",
    ],
}


class Seq2SeqValidityMetrics:
    def __init__(self, df, col_name):
        self.df = df
        self.col_name = col_name
        self.metrics = {metric_id: 0 for metric_id in METRICS_ERROR_MAP.keys()}
        self.eval = None

    def _validate(self, row):
        validator = SeqValidator()
        errors = validator.validate(
            row[self.col_name]
            .replace(" ", "")
            .replace("=>", " => ")  # this needs to be replace later to post process
        )
        # flatten the error codes from the METRICS_ERROR_MAP
        error_codes = [y for x in METRICS_ERROR_MAP.values() for y in x]
        for error_code in error_codes:
            row[error_code] = 0
        for metric_id in self.metrics.keys():
            row[metric_id] = 0
        updated_syn = False
        updated_func = False
        updated_align = False
        for error in set(errors):
            if error in error_codes:
                row[error] = 1
                metric_id = [
                    key
                    for key in METRICS_ERROR_MAP.keys()
                    if error in METRICS_ERROR_MAP[key]
                ]
                assert len(metric_id) == 1, f"Error code {error} is not unique"
                if metric_id[0] == "synthetic":
                    if not updated_syn:
                        self.metrics[metric_id[0]] += 1
                        updated_syn = True
                elif metric_id[0] == "functional":
                    if not updated_func:
                        self.metrics[metric_id[0]] += 1
                        updated_func = True
                elif metric_id[0] == "alignment":
                    if not updated_align:
                        self.metrics[metric_id[0]] += 1
                        updated_align = True
                row[metric_id[0]] = 1
        # now divide the errors to fit in the categories same as METRIC_ERROR_MAP

        return row

    def add_to_metric(self, name, value):
        for metric_id, metric_names in METRICS_ERROR_MAP.items():
            if name in metric_names:
                self.metrics[metric_id] = self.metrics[metric_id] + value
                return metric_id
        raise ValueError(f"Invalid error code: {name}")

    def evaluate(self):
        self.df = self.df.apply(self._validate, axis=1)
        self.eval = {
            metric_id: (metric_value / self.df.shape[0])
            for metric_id, metric_value in self.metrics.items()
        }
        return self.eval

    def compute(self):
        self.evaluate()
        validation_score = {key: -1 for key in self.metrics.keys()}
        if self.eval:
            validation_score = {key: 1 - self.eval[key] for key in self.metrics.keys()}
        return validation_score

    def save_df(self, save_path):
        self.df.to_csv(save_path, index=False)

    def visualize(self, save_path=None):
        """
        Visualizes the error distribution from METRICS_ERROR_MAP, excluding <unsupported-var-code>.
        """
        if self.eval is None:
            self.evaluate()

        # Filter out <unsupported-var-code> as requested
        # Sort metrics by value for better readability
        columns = [
            column for column in self.df.columns if column.startswith("<invalid")
        ]
        column_values = [sum(self.df[column].values) for column in columns]
        display_metrics = {
            column: value for column, value in zip(columns, column_values)
        }

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
                "##E:0:4:1:27## ${Cccc(C(=O)O)c(O)c(C(=O)O)c1} => ([O&X2&H1&!$(OC=*):1].[C&X3:2](=O)[O&X2&H1])>>(*-[O&X2:1].[C&X3:2](=O)-*) => *Oc1c(C(*)=O)cc(C)cc1C(=O)O$",
                # "##E:0:8:6:83## ${Cc1cc(CC(=O)X)c(C(=O)Cl)cc1C} + {O=C1c2ccccc2C(=O)c2cc(-c3ccc(O)cc3O)c(-c3cc(O)cc(O)c3)cc21} => ([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*) => *Oc1cc(O)cc(-c2cc3c(cc2-c2ccc(O)cc2OC(=O)Cc2cc(C)c(C)cc2C(*)=O)C(=O)c2ccccc2C3=O)c1$",
                # "##E:0:8:6:89## ${Cc1cc(CC(=O)X)c(C(=O)Cl)cc1C} + {O=C1c2ccccc2C(=O)c2cc(-c3ccc(O)cc3O)c(-c3cc(O)cc(O)c3)cc21} => ([C&X3:1](=O)[O&X2&H1,Cl,Br].[C&X3:2](=O)[O&X2&H1,Cl,Br]).([O,S;X2;H1;!$([O,S]C=*):3].[O,S;X2;H1;!$([O,S]C=*):4])>>(*-[C&X3:1]=O.[C&X3:2](=O)-[O,S;X2;!$([O,S]C=*):3].[O,S;X2;!$([O,S]C=*):4]-*) => *Oc1cc(O)cc(-c2cc3c(cc2-c2ccc(O)cc2OC(=O)Cc2cc(C)c(C)cc2C(*)=O)C(=O)c2ccccc2C3=O)c1$",
            ]
        }
    )
    metrics = Seq2SeqValidityMetrics(df, "sequence")
    score = metrics.compute()
    metrics.visualize(save_path="./metrics_plot.png")
    metrics.save_df(save_path="./metrics_data.csv")
    print("score >>", score)
