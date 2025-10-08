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
        open(smiles_path.parent / f"matcher_{smiles_path.stem}.pkl", "rb")
    )

    # Run property prediction
    csv_fname = smiles_path.parent / f"{smiles_path.stem}.csv"
    df = pd.read_csv(csv_fname)
    df["SMILES"] = df["id"].apply(lambda x: payload[x]["SMILES"])
    df["xtb_coordinates_path"] = df["id"].apply(
        lambda x: payload[x]["xtb_coordinates_path"]
    )
    df.to_csv(csv_fname)

    df = pd.read_csv(csv_fname)
    good_props = df[(df["s1"] < 2.8) & (df["s1"] > 2.6) & (df["delta"] < 0.2)]
    good_idx = [int(item[4:]) for item in good_props["id"]]
    print(
        f"There are {len(good_idx)} molecules that passed the property check and they represent {(100 * len(good_idx) / len(smiles_list)):.2f} % of the dataset."
    )
    data["statistics"]["num_pass_property"] = len(good_idx)
    print(data["statistics"])
    df = df[(df["s1"] < 2.8) & (df["s1"] > 2.6) & (df["delta"] < 0.2)]
    df.to_csv(csv_fname)
    with open(smiles_path, "wb") as file:
        pickle.dump(data, file)
