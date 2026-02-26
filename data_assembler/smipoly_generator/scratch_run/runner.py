import sys
import os
import pandas as pd
import logging
sys.path.append(os.path.join(os.path.dirname(__file__),'src'))

# the MAX_WORKERS will be set to None for all CPU core usage
MAX_WORKERS=22

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s -%(levelname)s -%(message)s'
)

from smipoly.smip import polg, monc

def filter_value(value):
    value0 = value.split(">>")[0]
    if value0 == None or value0.strip == "":
        raise ValueError(f"Wrong value found while parsing : {value0}")
    return value0 

if __name__ == "__main__":
    # this section will be used if the monomer classification is not done prior to this operation
#    df = pd.read_csv("../smiles_only.csv")
#    logging.info(f"[+] Loaded the Dataframe to memory: Shape {df.shape}")
#    cls_monomer_df = monc.moncls(df, smiColn="SMILES", minFG=2, maxFG=4, dsp_rsl=True) 
#    logging.info(f"[+] Classified Monomers: Shape {df.shape}")
#    cls_monomer_df.to_csv("monomer_classified_df.csv") 
#    logging.info("[+] saved the classification result at >>> monomer_classified_df.csv")
    # this section will be used if the monomer classification is already done
    # cls_monomer_df = pd.read_csv("monomer_classified_df.csv")
    cls_monomer_df = pd.read_parquet("test_df_5.parquet")
    # testing with lower thereshold as the larger set was failing
    # thresh = 25_000 # 1 / 10 for the dataset
    #  thresh_str = "25K" # 1 / 10 for the dataset
    #  cls_monomer_df = cls_monomer_df.iloc[:thresh, :] 
    logging.info(f"[+] Loaded the Classified Dataframe to memory: Shape {cls_monomer_df.shape}")
    # implementing multi threaded
    res_mt = polg.biplym(cls_monomer_df, targ=['all'], dsp_rsl=True)
    # implementing single threaded
    # res_mt = polg.biplym(cls_monomer_df, targ=['all'], dsp_rsl=True)
    # res_mt.to_parquet(f"test_5_polymer_output.csv") 
    # logging.info("[+] Completed The polymerizations")
    # logging.info(f"[+] saved the result at >>> test_5_polymer_output.csv")
