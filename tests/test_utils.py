import os
import json
from pathlib import Path

def _get_test_data_file_path(name) -> str:
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'test_data', name)

def get_validators_from_test_data_file(filename) -> list[dict]:
    path = _get_test_data_file_path(filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def get_test_data_list() -> list[list[dict]]:
    test_data_dir = Path(__file__).parent / 'test_data'
    files = [file for file in test_data_dir.iterdir() if file.is_file()]
    data = []
    for f in files:
        with open(f, 'r', encoding='utf-8') as f:
            data.append(json.load(f))
    return data
