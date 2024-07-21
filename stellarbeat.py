import json
import os
from requests import get
from platformdirs import user_cache_dir
import fbas

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
    cache_dir = user_cache_dir('python-fbas', 'SDF', ensure_exists=True)
    path = os.path.join(cache_dir, 'validators.json')
    if update:
        print(f"Updating data at {path}")
        validators = _fetch_from_url()
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(validators, f)
    else:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                validators = json.load(f)
        except FileNotFoundError:
            validators = _fetch_from_url()
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(validators, f)
    return validators

validators = get_validators()

def org_of_qset(qs : fbas.QSet):
    home_domains = {validators[v]['homeDomain'] for v in qs.validators}
    if len(home_domains) == 1:
        return home_domains.pop()
    else:
        return None