import argparse
import pickle
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles_path", type=str)
    args = parser.parse_args()

    smiles_path = Path(args.smiles_path)
    data = pickle.load(open(smiles_path, "rb"))
    smiles_list = data["smiles"]

    payload = pickle.load(
        open(smiles_path.parent / f"filtering_{smiles_path.stem}.pkl", "rb")
    )

    # Run property prediction
    csv_fname = smiles_path.parent / f"{smiles_path.stem}_filtering.csv"
    df = pd.read_csv(csv_fname, index_col=0)
    df["SMILES"] = df["molecule_id"].apply(lambda x: payload[x]["SMILES"])
    df["basic_coordinates_path"] = df["molecule_id"].apply(
        lambda x: payload[x]["basic_coordinates_path"]
    )
    df.to_csv(csv_fname)

    df = pd.read_csv(csv_fname, index_col=0)
    good_props = df[(df["vs1"] > 2.6) & (df["vdelta"] < 0.3)]
    good_idx = [int(item[4:]) for item in good_props["molecule_id"]]
    print(
        f"There are {len(good_idx)} molecules that passed the property check and they represent {(100 * len(good_idx) / len(smiles_list)):.2f} % of the dataset."
    )
    data["statistics"]["num_pass_property"] = len(good_idx)
    print(data["statistics"])
    df = df[(df["vs1"] > 2.6) & (df["vdelta"] < 0.3) & (df["vdelta"] >= 0)]
    df.to_csv(csv_fname)
    data["smiles"] = df["SMILES"].tolist() # Filter-out the bad SMILES.
    data["molecule_id"] = df["molecule_id"].tolist()
    with open(smiles_path, "wb") as file:
        pickle.dump(data, file)
