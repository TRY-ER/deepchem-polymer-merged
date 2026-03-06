import pandas as pd


def main(path):
    df = pd.read_csv(path)
    for row in df.iterrows():
        overall_value = 0
        for column in df.columns:
            if column in ["synthetic", "functional", "alignment"]:
                value = row[1][column]
                if column == "functional" and value == 0:
                    print("row >>", row)
                overall_value += int(value)
        if overall_value == 0:
            print("complete valid data found >>", row)
        # print("overall value >>", overall_value)


if __name__ == "__main__":
    path = "validity_metrics.csv"
    main(path)
