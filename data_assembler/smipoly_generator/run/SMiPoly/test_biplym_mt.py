import sys
import os
import time
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from smipoly.smip import polg

def create_dummy_dataframe(n_rows=10):
    mon_cols = list(polg.mon_dic.keys())
    
    # Simple valid SMILES
    smiles_list = ["CC=C", "C=CC(=O)O", "COC(=O)C=C", "c1ccccc1C=C"] * (n_rows // 4 + 1)
    
    data = {}
    data['smip_cand_mons'] = smiles_list[:n_rows]

    for col in mon_cols:
        data[col] = [True] * n_rows 
        
    df = pd.DataFrame(data)
    return df

def test_biplym_mt():
    print(f"Using polg from: {polg.__file__}", flush=True)
    
    print("Creating dummy data...", flush=True)
    df = create_dummy_dataframe(n_rows=50) # Larger data to overcome overhead
    
    print(f"Dataframe shape: {df.shape}", flush=True)
    print(f"Target classes: {list(polg.Ps_classL.keys())}", flush=True)
    
    print("\nRunning biplym (single threaded)...", flush=True)
    start = time.time()
    try:
        res_st = polg.biplym(df, targ=['all'], dsp_rsl=True)
    except Exception as e:
        print(f"biplym failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    time_st = time.time() - start
    print(f"Single threaded time: {time_st:.4f}s", flush=True)
    
    print("\nRunning biplym_mt (multithreaded)...", flush=True)
    start = time.time()
    try:
        # Use more workers or default (None = num_cpus * 5)
        res_mt = polg.biplym_mt(df, targ=['all'], dsp_rsl=True, max_workers=None)
    except Exception as e:
        print(f"biplym_mt failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return

    time_mt = time.time() - start
    print(f"Multithreaded time: {time_mt:.4f}s", flush=True)
    
    # Verify results
    print(f"\nST rows: {len(res_st)}", flush=True)
    print(f"MT rows: {len(res_mt)}", flush=True)
    
    if len(res_st) != len(res_mt):
        print("Result lengths differ!", flush=True)
        raise AssertionError("Result lengths differ!")

    # Check if contents are similar
    res_st['polym_str'] = res_st['polym'].astype(str)
    res_mt['polym_str'] = res_mt['polym'].astype(str)
    
    st_polyms = sorted(res_st['polym_str'].tolist())
    mt_polyms = sorted(res_mt['polym_str'].tolist())
    
    if st_polyms == mt_polyms:
        print("Polymers match exactly!", flush=True)
    else:
        print("Polymers content mismatch!", flush=True)
        raise AssertionError("Polymers content mismatch!")

    res_st.to_csv("test_df_st.csv")
    print("\nVerification passed!", flush=True)
    with open("test_result.txt", "w") as f:
        f.write("Verification passed!\n")

if __name__ == "__main__":
    test_biplym_mt()
