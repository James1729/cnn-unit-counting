import json 

from pathlib import Path
from typing import Dict
from glob import glob


def load_json(filename: str) -> Dict:
    with open(filename) as f:
        return json.load(f)

def write_json(data: Dict, filename: str) -> None:
    with open(filename, 'w') as f:
        json.dump(data, f)

def get_unit_references(faction: str, colour: str, unit: str) -> Dict:
    return load_json(filename=f"raw_stock_images/library_references/{faction}/{colour}/{unit}/references.json")

def get_background_paths(folder: str) -> Dict:
    return glob(f"raw_stock_images/processed/backgrounds/{folder}/*")