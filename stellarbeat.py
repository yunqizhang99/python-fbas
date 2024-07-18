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

def create_fake_nodes(n) -> list:
    """
    Create n fake nodes and put them in a list
    """
    all_fake_nodes = []
    res = []
    for i in range(0, n):
        pk = "MALICIOUS" + str(i)
        all_fake_nodes.append(pk)
    for i in range(0, n):
        pk = "MALICIOUS" + str(i)
        cur_node_validators = all_fake_nodes.copy()
        cur_node_validators.remove(pk)
        qs = {"threshold":n/2, "validators":cur_node_validators, "innerQuorumSets":[]}
        cur_node = {"publicKey": pk, "quorumSet": qs, "isValidator": True}
        res.append(cur_node)
    return res, all_fake_nodes

def _fetch_with_fake_nodes() -> list:
    """
    Get data from stellarbeat, filter out non-validator nodes, and return a list of dicts with two keys: 'publicKey' and 'quorumSet'.
    """
    url = "https://api.stellarbeat.io/v1/node"
    response = get(url, timeout=5)
    num_fake_nodes = 10
    fake_nodes, fake_pks = create_fake_nodes(num_fake_nodes)
    if response.status_code == 200:
        data = response.json()
        for item in data:
            if item['host'] != None and "satoshipay" in item['host']:
                traitor_qs = {"threshold":num_fake_nodes/2 + 1, "validators":fake_pks, "innerQuorumSets":[]}
                item['quorumSet'] = traitor_qs
        while len(fake_nodes) > 0:
            data.append(fake_nodes.pop())
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