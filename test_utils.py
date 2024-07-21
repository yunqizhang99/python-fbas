import os
import json

def get_test_data_file_path(name):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'test_data', name)

def get_validators_from_file(path) -> list[dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return {v['publicKey'] : v for v in json.load(f)}
