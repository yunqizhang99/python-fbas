import json
import os
from typing import Optional
from requests import get
from platformdirs import user_cache_dir
import python_fbas.fbas as fbas

STELLARBEAT_URL = "https://api.stellarbeat.io/v1/node"
validators = dict() # will be populated with data from stellarbeat at module initialization

def _process_validators(json_data) -> list[dict]:
        # raise if json_data is not a list:
        if not isinstance(json_data, list):
            raise ValueError("json_data must be a list")
        result = dict()
        for v in json_data:
            match v:
                case {'publicKey': _, 'quorumSet': _, 'isValidator': True}:
                    if v['publicKey'] in result:
                        raise ValueError(f"Duplicate validator found: {v['publicKey']}")
                    result |= {v['publicKey'] : v}
                case _:
                    pass
        return result

def _fetch_from_url() -> dict:
    """
    Get data from stellarbeat and filter out non-validator nodes
    """
    response = get(STELLARBEAT_URL, timeout=5)
    if response.status_code == 200:
        return _process_validators(response.json())
    else:
        response.raise_for_status()

def get_validators(update=False) -> dict:
    """When update is true, fetch new data from stellarbeat and update the file in the cache directory."""
    cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
    path = os.path.join(cache_dir, 'validators.json')
    if update:
        print(f"Updating data at {path}")
        _validators = _fetch_from_url()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(_validators, f)
    else:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                _validators = json.load(f)
        except FileNotFoundError:
            _validators = _fetch_from_url()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(_validators, f)
    return _validators

validators = get_validators()

def org_of_qset(qs : fbas.QSet) -> Optional[str]:
    """If this qset represents an arganization, i.e. all validators have the same homeDomain, return that domain. Otherwise return None."""
    home_domains = {validators[v]['homeDomain'] for v in qs.validators}
    if len(home_domains) == 1:
        return home_domains.pop()
    else:
        return None