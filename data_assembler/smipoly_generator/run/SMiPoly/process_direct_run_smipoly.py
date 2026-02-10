import cudf


def process_direct_run_smipoly(
    df_path, processed_path, summary_path="summary.csv", histogram_path="histogram.png"
):
    # Load data from CSV files
    df = cudf.read_csv(df_path)

    # Perform data cleaning and preprocessing
    df = df.dropna()
    df = df.drop_duplicates(subset=["polym"])

    # Perform data analysis and visualization
    df.describe().to_csv(summary_path)

    # describe polym column
    df["polym"].describe().to_csv("polym_summary.csv")
    df.hist().savefig(histogram_path)

    # Save processed data to Parquet file
    df.to_parquet(processed_path)


if __name__ == "__main__":
    input_path = ""
    processed_path = ""
    process_direct_run_smipoly(input_path, processed_path)
