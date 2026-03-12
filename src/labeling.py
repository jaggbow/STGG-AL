import argparse
import pickle
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str)
    parser.add_argument("--old_master_path", type=str)
    args = parser.parse_args()

    smiles_path = Path(args.smiles_path)
    old_master_path = Path(args.old_master_path)
    smiles_id = smiles_path.stem
    master_id = int(old_master_path.stem.split("_")[-1])

    new_master = pd.read_csv(args.old_master_path, index_col=0)
    payload = pickle.load(
        open(smiles_path.parent / f"labeling_{smiles_id}.pkl", "rb")
    )
    # Run the labeling
    csv_labeling = smiles_path.parent / f"{smiles_id}_labeling.csv"
    csv_filtering = smiles_path.parent / f"{smiles_id}_filtering.csv"
    csv_gaussian = smiles_path.parent / f"{smiles_id}_gaussian.csv"

    df_labeling = pd.read_csv(csv_labeling, index_col=0)
    if "vs1_xtb" in df_labeling:
        del df_labeling["vs1_xtb"]
    if "vdelta_xtb" in df_labeling:
        del df_labeling["vdelta_xtb"]

    df_labeling["SMILES"] = df_labeling["molecule_id"].apply(lambda x: payload[x]["SMILES"])
    df_labeling["xtb_coordinates_path"] = df_labeling["molecule_id"].apply(
        lambda x: payload[x]["xtb_coordinates_path"]
    )
    df_labeling["vs1_xtb_gt"] = df_labeling["molecule_id"].apply(
        lambda x: payload[x]["vs1_xtb"]
    )
    df_labeling["vdelta_xtb_gt"] = df_labeling["molecule_id"].apply(
        lambda x: payload[x]["vdelta_xtb"]
    )
    if csv_gaussian.exists():
        df_gaussian = pd.read_csv(csv_gaussian, index_col=0)
        df_gaussian = df_gaussian.rename(columns={"vs1": "vs1_labeling", "vdelta": "vdelta_labeling"})
        df_gaussian = df_gaussian[["molecule_id", "SMILES", "vs1_gt", "vdelta_gt"]]
        df_labeling = df_labeling.merge(df_gaussian, how="left", on=["molecule_id", "SMILES"])

    df_labeling = df_labeling.rename(columns={"vs1": "vs1_labeling", "vdelta": "vdelta_labeling"})

    df_filtering = pd.read_csv(csv_filtering, index_col=0)
    df_filtering = df_filtering.merge(df_labeling, how="right", on=["molecule_id", "SMILES"])

    df_filtering["molecule_id"] = smiles_id + "_" + df_filtering["molecule_id"]
    df_filtering["target_core"] = 1
    df_filtering["mmff_coordinates_path"] = df_filtering["basic_coordinates_path"].apply(lambda x: (Path(x).parent / "mmff.xyz").absolute().as_posix())
    
    data = pickle.load(open(smiles_path, "rb"))
    data["smiles"] = df_filtering.SMILES.tolist()
    data["molecule_id"] = df_filtering["molecule_id"].tolist()
    data["statistics"]["num_pass_labeling"] = len(data["smiles"])
    with open(smiles_path, "wb") as file:
        pickle.dump(data, file)    
    new_master = pd.concat([new_master, df_filtering], ignore_index=True)
    new_master.to_csv(old_master_path.parent / f"master_{master_id+1}.csv") 
