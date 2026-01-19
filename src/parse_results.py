import os, sys
import glob
from pathlib import Path
import shutil
import pickle
import pandas as pd
import argparse
au2ev = 27.2114

#s1_slope = 0.9527960191042272
#s1_intercept = -0.40221058
#t1_slope = 0.889516986
#t1_intercept = -0.03654902

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_filtering", required=True, type=str)
    parser.add_argument("--csv_labeling", required=True, type=str)
    parser.add_argument("--csv_master", required=True, type=str)
    parser.add_argument("--csv_gaussian", required=False, default="", type=str)
    parser.add_argument("--smiles_path", required=True, type=str)
    parser.add_argument("--generator_data_path", type=str)
    parser.add_argument("--property_predictor_data_path", type=str)
    args = parser.parse_args()
   
    df_filtering = pd.read_csv(args.csv_filtering, index_col=0)
    df_labeling = pd.read_csv(args.csv_labeling, index_col=0)
    if args.csv_gaussian != "":
        df_gaussian = pd.read_csv(args.csv_gaussian, index_col=0)

    

    smiles_data = pickle.load(open(args.smiles_path, "rb"))
   
    # Move old data and update it with new entries
    generator_data_path = Path(args.generator_data_path)
    data = pd.read_csv(generator_data_path)
    
    subset = gaussian_df[["smiles", "aS1", "adelta", "vS1", "vdelta"]]
    subset["target_core"] = 1
    subset["molecule_id"] = gaussian_df["name"].apply(lambda x: f"{csv_path.stem}_{x}")
    result_df = pd.concat([data, subset], ignore_index=True)
    result_df = result_df.drop("Unnamed: 0", axis=1)
    print(result_df.describe())
    shutil.move(args.generator_data_path, csv_path.parent / f"generator_data_{csv_path.stem}.csv")
    result_df.to_csv(generator_data_path)

    property_predictor_data_path = Path(args.property_predictor_data_path)
    data = pd.read_csv(property_predictor_data_path)
    
    subset = gaussian_df[["smiles", "aS1", "adelta", "vS1", "vdelta"]]
    subset["xtb_coordinates_path"] = gaussian_df["name"].apply(lambda x: coordinates_dir / f"{x[4:]}")
    subset["molecule_id"] = gaussian_df["name"].apply(lambda x: f"{csv_path.stem}_{x}")
    result_df = pd.concat([data, subset], ignore_index=True)
    result_df = result_df.drop("Unnamed: 0", axis=1)
    print(result_df.describe())
    shutil.move(property_predictor_data_path, csv_path.parent / f"property_predictor_data_{csv_path.stem}.csv")
    result_df.to_csv(property_predictor_data_path)
