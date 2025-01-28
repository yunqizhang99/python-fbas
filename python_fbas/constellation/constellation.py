import json
import logging
import subprocess
import random
from collections import defaultdict
from itertools import combinations, combinations_with_replacement, product
from typing import Optional
import networkx as nx
from python_fbas.fbas_graph import FBASGraph

def load_regular_fbas_from_file(file_name:str) -> dict:
    """
    Load a regular FBAS from a file. The file must contain a JSON dictionary where each key is an
    organization and the value is a pair consisting of an integer threshold and a list of
    organizations.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        return json.load(f)

def single_universe_to_regular(fbas: dict[str,int]) -> dict[str,tuple[int,list[str]]]:
    """
    Convert a single-universe regular FBAS to a regular FBAS.
    """
    orgs = list(fbas.keys())
    return {org: (fbas[org], orgs) for org in fbas}

def load_single_universe_regular_fbas_from_file(file_name:str) -> dict:
    """
    Load a single-universe, regular FBAS from a file. The file must contain a JSON dictionary where
    each key is an organization and the value is an integer threshold.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        thresholds = json.load(f)
    return single_universe_to_regular(thresholds)

def random_single_universe_regular_fbas(n:int, low:int, high:int) -> dict:
    """
    Generate a random single-universe regular FBAS with n organizations and thresholds between low and high.
    """
    return {f"O_{i}": random.randint(low, high) for i in range(1, n+1)}

def load_survey_graph(file_name) -> nx.Graph:
    """
    Load the overlay graph from Stellar survey data.
    """
    with open(file_name, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    g = nx.Graph()
    for v, peers in json_data.items():
        inbound = set(peers['inboundPeers'].keys())
        outbound = set(peers['outboundPeers'].keys())
        for peer in inbound | outbound:
            g.add_edge(v, peer)
    return g

# Constellation handles a specific type of FBAS, where each validator requires agreement from a
# threshold among a set of organizations, and each organization runs 3 nodes with a threshold of 2.
def regular_fbas_to_fbas_graph(regular_fbas) -> FBASGraph:
    """
    Convert a regular FBAS to a FBASGraph. A regular FBAS consists of a set of organizations where
    each organization O requires agreement from a threshold t_O among a set of organizations S_O.
    """
    assert isinstance(regular_fbas, dict)
    for org in regular_fbas:
        assert isinstance(org, str)
        # org must be map to a pair (threshold, list of organizations):
        assert isinstance(regular_fbas[org], tuple)
        assert isinstance(regular_fbas[org][0], int)
        assert regular_fbas[org][0] > 0
        assert isinstance(regular_fbas[org][1], list)
        for o in regular_fbas[org][1]:
            assert isinstance(o, str)
            assert o in regular_fbas

    # now build the FBASGraph
    def org_inner_qset(o):
        return {'threshold': 2, 'validators': [f'{o}_1', f'{o}_2', f'{o}_3'], 'innerQuorumSets': []}
    fbas_graph = FBASGraph()
    for org in regular_fbas:
        match regular_fbas[org]:
            case (threshold, orgs):
                qset = {'threshold': threshold, 'validators': [],'innerQuorumSets': [org_inner_qset(o) for o in orgs]}
                for n in range(1, 4):
                    fbas_graph.update_validator(f'{org}_{n}', qset)
    return fbas_graph

def parse_output(output:str) -> list[dict[int,int]]:
    """
    Parse the output of the optimal partitioning algorithm.

    A partition is represented by a string; for example, "1.1.2.2.2.3.5|1.4.4|2.2.3" means [{1:2, 2:3, 3:1, 5:1}, {1:1, 4:2}, {2:2, 3:1}].
    In the output of the C program, the optimal partition appears on the last but 2 line.
    """
    # get the last but 2 line:
    lines = output.splitlines()
    partition = lines[-2]
    partitions = partition.split('|')
    result = []
    for p in partitions:
        d:dict[int,int] = defaultdict(int)
        for x in p.split('.'):
            d[int(x)] += 1
        result.append(d)
    return result

def compute_clusters(regular_fbas:dict, min_cluster_size:int = 1, max_num_clusters:Optional[int]=None) -> list[set[str]]:
    """
    Determines the Constellation clusters by calling the C implementation of the optimal-partitioning algorithm.
    The command 'optimal_cluster_assignment' must be in the PATH.

    This only uses the threshold of each organization, not its universe (which is implicitely assumed to be all organizations).
    """
    n_orgs = len(regular_fbas.keys())
    threshold_multiplicity:dict[int,int] = defaultdict(int)
    for org in regular_fbas:
        match regular_fbas[org]:
            case (t, _):
                threshold_multiplicity[n_orgs - t + 1] += 1
    # build the command-line arguments:
    arg_pairs = [[threshold_multiplicity[t], t] for t in threshold_multiplicity.keys()]
    args = [x for sublist in arg_pairs for x in sublist] # flatten the list
    # # if there are more than n_limit organizations, limit the mimimum cluster size to n_orgs/size_denom and the max number of clusters to max_clusters
    # n_limit = 12
    # size_denom = 5
    # min_cluster_size = int(n_orgs/size_denom)+1 if n_orgs > n_limit else 1
    # # limit the number of clusters to 4:
    # max_clusters = 4 if n_orgs > n_limit else n_orgs
    args = args + [min_cluster_size, max_num_clusters if max_num_clusters else n_orgs]
    # obtain the optimal partition:
    logging.debug("calling optimal_cluster_assignment with args: %s", args)
    output = subprocess.run(['optimal_cluster_assignment'] + [str(x) for x in args], capture_output=True, text=True, check=True)
    partition = parse_output(output.stdout) # TODO error handling
    # now assign organizations to the clusters
    threshold_map:dict[int,list[str]] = {} # map each threshold to the set of organizations that have it
    for org in regular_fbas:
        match regular_fbas[org]:
            case (t, _):
                t = n_orgs - t + 1
                if t not in threshold_map:
                    threshold_map[t] = []
                threshold_map[t].append(org)
    # sort the blocking thresholds in decreasing order:
    index_of_threshold = {t:i for i,t in enumerate(sorted(threshold_multiplicity.keys(), reverse=True))}
    clusters:list[set[str]] = [set() for _ in partition] # start with empty cluters
    for t, orgs in threshold_map.items():
        for i, part in enumerate(partition):
            n = part.get(index_of_threshold[t]+1, 0)
            clusters[i] |= set(orgs[:n])
            orgs = orgs[n:]
    return clusters

def clusters_to_overlay(clusters:list[set[str]]) -> nx.Graph:
    """
    Given a list of clusters, return the Constellation overlay graph.

    Each organization O has 3 nodes O_0, O_1, and O_2 that are fully connected.
    If two organizations are in different clusters, then each node in the first organization is connected to each node in the second organization.
    If two organizations O and O' are in the same cluster, then O_i is connected to O'_{i+1 mod 3} and O'{i+2 mod 3}.
    """
    g = nx.Graph()
    orgs = set.union(*clusters)
    # map orgs to their cluster:
    cluster_map = {org: i for i, cluster in enumerate(clusters) for org in cluster}
    for org in orgs:
        # first, connect the org's nodes in a complete graph:
        # for all combinations i, j in {0, 1, 2} (where order does not matter), connect org_i to org_j:
        for i, j in combinations(range(3), 2):
            g.add_edge(f'{org}_{i}', f'{org}_{j}')
        # now connect the org's nodes to the nodes of other organizations in the same cluster:
        cluster = clusters[cluster_map[org]]
        for other_org in cluster:
            if org == other_org:
                continue
            if cluster_map[other_org] == cluster_map[org]:
                # same cluster:
                for i in range(3):
                    g.add_edge(f'{org}_{i}', f'{other_org}_{(i+1)%3}')
                    g.add_edge(f'{org}_{i}', f'{other_org}_{(i+2)%3}')
    # now create the inter-cluster edges:
    for c1, c2 in combinations(clusters, 2):
        max_c,min_c = (list(c1),list(c2)) if len(c1) > len(c2) else (list(c2),list(c1))
        for n, org in enumerate(max_c):
            other = min_c[n%len(min_c)]
            for i,j in product(range(3), range(3)):
                g.add_edge(f'{org}_{i}', f'{other}_{j}')
    return g

def constellation_overlay(regular_fbas:dict) -> nx.Graph:
    """
    Given a regular FBAS, return the Constellation overlay graph.
    """
    # first we transform the regular fbas into a single-universe regular fbas:
    clusters = compute_clusters(regular_fbas)
    logging.info("got %s clusters", len(clusters))
    return clusters_to_overlay(clusters)

    