# data/dataset_loader.py

import os
import pandas as pd

def load_info_txt(data_root: str, include_normal: bool = True) -> pd.DataFrame:
    """Load and preprocess the Info.txt file."""
    info_path = os.path.join(data_root, "Info.txt")
    info = pd.read_csv(info_path, sep=" ")

    if 'Unnamed: 7' in info.columns:
        info.drop(columns=['Unnamed: 7'], inplace=True)

    if include_normal:
        info['SEVERITY'] = info['SEVERITY'].fillna('N')
    else:
        info = info.dropna(subset=['SEVERITY'])

    # Optional rebalancing (drop excess N if overrepresented)
    labels, counts = info['SEVERITY'].value_counts().index, info['SEVERITY'].value_counts().values
    if include_normal and len(counts) == 3 and counts[2] > max(counts[:2]):
        drop_idx = info['SEVERITY'].mask(lambda x: x != 'N').dropna().sample(counts[2] - max(counts[:2])).index
        info = info.drop(drop_idx)

    return info
