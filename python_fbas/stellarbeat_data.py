"""
Module to fetch data from stellarbeat.io
"""

import json
import os
import logging
from requests import get
from platformdirs import user_cache_dir

STELLARBEAT_URL = "https://api.stellarbeat.io/v1/node"

def _fetch_from_url() -> list[dict]:
    """
    Get data from stellarbeat
    """
    logging.info("Fetching data from stellarbeat.io")
    response = get(STELLARBEAT_URL, timeout=5)
    if response.status_code == 200:
        return response.json()
    response.raise_for_status()
    raise IOError("Failed to fetch data from stellarbeat")

def get_validators(update=False) -> list[dict]:
    """
    When update is true, fetch new data from stellarbeat and update the file in the cache directory.
    """
    cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
    path = os.path.join(cache_dir, 'stellarbeat.json')
    def update_cache_file(_validators):
        logging.info("Writing stellarbeat data at %s", path)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_validators, f)
    if update:
        print(f"Updating cache at {path}")
        _validators = _fetch_from_url()
        update_cache_file(_validators)
    else:
        try:
            logging.info("Reading stellarbeat data from %s", path)
            with open(path, 'r', encoding='utf-8') as f:
                _validators = json.load(f)
        except FileNotFoundError:
            logging.info("Cache file not found")
            _validators = _fetch_from_url()
            update_cache_file(_validators)
    return _validators
