# data/dataset_loader.py
# Copyright (C) 2025 ibrar-syed <syed.ibraras@gmail.com>
# This file is part of the Cancer Detection Project.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

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
