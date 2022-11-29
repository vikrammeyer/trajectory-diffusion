import json
import pathlib
import pickle

def write_obj(obj, filename: pathlib.Path):
    with open(filename, "wb") as f:
        pickle.dump(obj, f)

def read_file(filename: pathlib.Path):
    with open(filename, "rb") as f:
        return pickle.load(f)

def write_metadata(cfg, dataset_folder: pathlib.Path):
    meta_file = dataset_folder / "metadata.json"
    metadata = json.dumps(cfg.__dict__)
    meta_file.write_text(metadata)
