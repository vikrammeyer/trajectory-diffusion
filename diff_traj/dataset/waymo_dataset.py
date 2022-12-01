import pathlib
import torch
import math
import numpy as np

import json

def clamp(n, smallest=-1, largest=1):
    return max(smallest, min(n, largest))

class WaymoDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_folder):
        super(WaymoDataset).__init__()

        dataset_folder = pathlib.Path(dataset_folder)

        self.traj = []

        json_files = dataset_folder.glob("*.json")

        for file in json_files:
            self.parse_json_file(file)

    def parse_json_file(self, file):
        data = json.loads(file.read_text())

