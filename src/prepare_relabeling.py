import argparse
import pickle
from pathlib import Path

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--master_path", type=str)
    args = parser.parse_args()

    old_master_path = Path(args.old_master_path)
    master_id = int(old_master_path.stem.split("_")[-1])
    
    master = pd.read_csv(args.old_master_path, index_col=0)

    new_master.to_csv(old_master_path.parent / f"master_{master_id+1}.csv") 
