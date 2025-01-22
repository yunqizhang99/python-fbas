import json
import networkx as nx
from python_fbas.fbas_graph import FBASGraph

def load_survey_graph(file_name) -> nx.Graph:
    with open(file_name, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    g = nx.Graph()
    for v, peers in json_data.items():
        inbound = set(peers['inboundPeers'].keys())
        outbound = set(peers['outboundPeers'].keys())
        for peer in inbound | outbound:
            g.add_edge(v, peer)
    return g

def symmetric_fbas_to_fbas_graph(symmetric_fbas) -> FBASGraph:
    """
    Convert a symmetric FBAS to a FBASGraph.
    A symmetric FBAS is just a threshold and a list of organizations.
    We assume that all organizations run 3 nodes.
    """
    match symmetric_fbas:
        case (threshold, orgs):
            if  not (isinstance(orgs, list) and isinstance(threshold, int) and threshold > 0 and threshold <= len(orgs)):
                raise ValueError("Invalid symmetric FBAS format")
            fbas_graph = FBASGraph()
            def org_qset(o):
                return {'threshold': 2, 'validators': [f'{o}_1', f'{o}_2', f'{o}_3'],'innerQuorumSets': []}
            qset = {'threshold': threshold, 'validators': [],'innerQuorumSets': [org_qset(o) for o in orgs]}
            for o in orgs:
                for n in range(1, 4):
                    fbas_graph.update_validator(f'{o}_{n}', qset)
            return fbas_graph
        case _:
            raise ValueError("Invalid symmetric FBAS format")
        
def single_universe_fbas_to_fbas_graph(single_universe_fbas) -> FBASGraph:
    """
    Convert a single-universe FBAS to a FBASGraph.
    A single universe FBAS is just list of organizations, each with its own threshold.
    We assume that all organizations run 3 nodes.
    """
    raise NotImplementedError("single-universe FBAS to FBASGraph conversion is not yet implemented")