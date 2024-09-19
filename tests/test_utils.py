import os
import json

def _get_test_data_file_path(name) -> str:
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'test_data', name)

def get_validators_from_test_data_file(filename) -> list[dict]:
    path = _get_test_data_file_path(filename)
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
