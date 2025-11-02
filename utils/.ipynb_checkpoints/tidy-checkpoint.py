import json 

from typing import Dict


def load_json(filename: str) -> Dict:
    with open(filename) as f:
        return json.load(f)
