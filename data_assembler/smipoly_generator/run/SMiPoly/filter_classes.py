import cudf

# available polymer classes (* marked relevant polymer pairs)
# -------------------------
# {"polyolefin": 11,
# "polyester": 6,
# "polyether": 12, *
# "polyamide": 2, *
# "polyimide": 8, *
# "polyurethane": 19, *
# "polyoxazolidone": 23}

filter_classes = {
    "polyamide": ["diCOOH", "diamin", "pridiamin"],
    "polyimide": ["dicAnhyd", "diamin", "pridiamin"],
    "polyuratheame": ["diNCO", "diol"],
    "polyester": ["lactone", "diCOOH", "diol"],
}

# value_type,count
# ________________
# Unknown,646269
# vinyl,145812
# cAnhyd,46781
# diol_b,43850
# pridiamin,41374 *
# diamin,36355 *
# lactone,20578 *
# cOle,16836
# epo,14296
# aminCOOH,13712
# diol,13609 *
# hydCOOH,8873
# hindPhenol,2271
# diNCO,2205 *
# diCOOH,951 *
# dicAnhyd,808 *
# diepo,118
# BzodiF,4
# sfonediX,4
# HCHO,1
# lactam,1
# CO,1

relevant_mon_classes = [
    "diCOOH",
    "diamin",
    "pridiamin",
    "dicAnhyd",
    "diol_b",
    "diNCO",
    "lactone",
]

# polymearization reaction pairs (* marked the relevant pairs)
# --------------------------------
# vinyl
# cOle
# vinyl_vinyl
# vinyl_cOle
# hydCOOH
# lactone * Ring Opening Polymerization [ Polyester ]
# hydCOOH_hydCOOH
# diol_CO
# diCOOH_diol * Poly Condensation [ Polyester ]
# cAnhyd_epo
# epo
# hindPhenol
# sfonediX_diol_b
# BzodiF_diol_b
# lactam
# aminCOOH
# diCOOH_diamin * Poly Condensation [ Polyamides ]
# aminCOOH_aminCOOH
# dicAnhyd_diamin * Poly Condensation [ Polyimides ]
# diNCO_diol * Poly Addition [ Polyurethanes ]
# diepo_diNCO

# Note
# ____
# There is no pair for pridiamin for some reason
# Look into that if we have missed something

def limit_df_to_threshold(df, column_names, column_map)

def main(input_df_path, output_path):
    df = cudf.read_csv(input_df_path)
    print("[+] Reading input file complete")

    # there are columns with name that in relevant_mon_classes
    # the columsn contain boolean true false values
    # keep rows with relevant monomer classes as true

    df = df[df[relevant_mon_classes].any(axis=1)]
    print("[+] Filtering monomer classes complete")

    # output summary of the modified dataframe into a summary.csv file
    df.describe().to_csv("summary.csv", index=False)
    print("[+] Summary file written")

    # output the modified dataframe into a parquet file
    df.to_parquet(output_path)
    print(f"[+] Output file written to {output_path}")


if __name__ == "__main__":
    input_df_path = "monomer_classified_df.csv"
    output_path = "relevant_filtered_df.parquet"
    main(input_df_path, output_path)
