import argparse
import pickle
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_data_dir", type=str)
    parser.add_argument("--master_path", type=str)

    args = parser.parse_args()

    OTHER_DATA_DIR = Path("../")
    
    df_master = pd.read_csv(args.master_path, index_col=0)
    df_labeling = df_master[~(df_master["vs1_gt"].isna() | df_master["vdelta_gt"].isna())]
    df_labeling = df_labeling[["molecule_id","SMILES","vs1_gt","vdelta_gt","vs1_xtb_gt","vdelta_xtb_gt","xtb_coordinates_path","mmff_coordinates_path","basic_coordinates_path"]]
    df_labeling = df_labeling.rename(columns={
        "vs1_gt": "vs1",
        "vdelta_gt": "vdelta",
        "vs1_xtb_gt": "vs1_xtb",
        "vdelta_xtb_gt": "vdelta_xtb",
    })
    df_labeling.to_csv(OTHER_DATA_DIR / "labeling.csv", mode="x")
