import argparse
import pickle
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_path", type=str)
    parser.add_argument("--relabel_master_path", type=str)
    args = parser.parse_args()

    
    relabel_master_path = Path(args.relabel_master_path)
    master_path = Path(args.master_path)
    
    master_id = int(master_path.stem.split("_")[-1])

    master = pd.read_csv(args.master_path, index_col=0)
    relabel_master = pd.read_csv(args.relabel_master_path, index_col=0)
    relabel_master = relabel_master.rename(columns={"vs1": "vs1_labeling", "vdelta": "vdelta_labeling"})
    df = master.merge(relabel_master, on='molecule_id', how='left', suffixes=('', '_new'))
    df['vs1_labeling'] = df['vs1_labeling_new']
    df['vdelta_labeling'] = df['vdelta_labeling_new']
    
    del df['vs1_labeling_new']
    del df['vdelta_labeling_new']
    
    df.to_csv(relabel_master_path)

