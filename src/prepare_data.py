import argparse
import pickle
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_data_dir", type=str)
    parser.add_argument("--master_path", type=str)
    parser.add_argument("--save_labeling", action="store_true")
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
    df_filtering = df_master[["molecule_id","SMILES","vs1_labeling","vdelta_labeling","vs1_xtb_gt","vdelta_xtb_gt","xtb_coordinates_path","mmff_coordinates_path","basic_coordinates_path"]]
    df_filtering = df_filtering.rename(columns={
        "vs1_labeling": "vs1",
        "vdelta_labeling": "vdelta",
        "vs1_xtb_gt": "vs1_xtb",
        "vdelta_xtb_gt": "vdelta_xtb",
    })
    df_filtering.loc[df_filtering['vs1'].isna(), 'vs1'] = df_master["vs1_gt"]
    df_filtering.loc[df_filtering['vdelta'].isna(), 'vdelta'] = df_master["vdelta_gt"]
    df_generator = df_master[["molecule_id","SMILES","target_core","vs1_labeling","vdelta_labeling"]]
    df_generator.loc[df_generator['vs1_labeling'].isna(), 'vs1_labeling'] = df_master["vs1_gt"]
    df_generator.loc[df_generator['vdelta_labeling'].isna(), 'vdelta_labeling'] = df_master["vdelta_gt"]
    df_generator = df_generator.rename(columns={
        "vs1_labeling": "vs1",
        "vdelta_labeling": "vdelta",
        "SMILES": "smiles"
    })
    df_generator.to_csv(Path(args.generator_data_dir) / "data.csv", mode="x")
    df_filtering.to_csv(OTHER_DATA_DIR / "filtering.csv", mode="x")
    if args.save_labeling:
        df_labeling.to_csv(OTHER_DATA_DIR / "labeling.csv", mode="x")
