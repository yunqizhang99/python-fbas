import json
import os
from requests import get
from platformdirs import user_cache_dir

def _fetch_from_url() -> list:
    """
    Get data from stellarbeat, filter out non-validator nodes, and return a list of dicts with two keys: 'publicKey' and 'quorumSet'.
    """
    url = "https://api.stellarbeat.io/v1/node"
    response = get(url, timeout=5)
    if response.status_code == 200:
        data = response.json()
        return [{'publicKey': node['publicKey'], 'quorumSet': node['quorumSet']}
            for node in data if node['isValidator']]
    else:
        response.raise_for_status()

def get_validators(update=False) -> list:
    cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
    path = os.path.join(cache_dir, 'validators.json')

    if update:
        print(f"Updating data at {path}")
        json_data = _fetch_from_url()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)
    else:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except FileNotFoundError:
            json_data = _fetch_from_url()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f)
    return json_data

def get_validators_from_file(path) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)