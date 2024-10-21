"""
Usage:
  script.py --check-assumptions --fbas=<fbas>
  script.py --find-parameter (--n_orgs=<n_orgs> | --fbas=<fbas>)
"""

import json
import os
import logging
import networkx as nx
from docopt import docopt
from python_fbas.fbas_generator import gen_symmetric_fbas
from python_fbas.fbas import QSet, FBAS

args = docopt(__doc__)

def check_failure_assumptions(g : nx.Graph, fbas : FBAS) -> bool:
    def node_satisfied(n):
        qset : QSet = fbas.qset_map[n]
        peers = g.neighbors(n)
        return qset.blocked(peers)
    return all(node_satisfied(n) for n in g.nodes())

def erdos_renyi(p, validators):
    g = nx.erdos_renyi_graph(len(validators), p)
    # map nodes to validators:
    mapping = {n : v for n, v in zip(g.nodes(), validators)}
    return nx.relabel_nodes(g, mapping)

# We want to find out whether, for a give parameter p, the failure assumptions of the fbas are satisfied with high probability:
def check_p(p, fbas):
    # we make a number of experiments and return true if 99% of them succeed:
    n = 5000
    success = 0
    for i in range(n):
        g = erdos_renyi(p, fbas.validators())
        if check_failure_assumptions(g, fbas):
            success += 1
    return success/n > 0.99

# Now we use a binary search between 0 and 1 to find the smallest p for which the failure assumptions are satisfied with high probability:
def find_p(fbas):
    p = 0.5
    step = 0.25
    while step > 0.01:
        if check_p(p, fbas):
            p -= step
        else:
            p += step
        step /= 2
    return p

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    if args['--find-parameter']:
        if args['--fbas']:
            file_path = os.path.expanduser(args['--fbas'].strip())
            with open(file_path, 'r', encoding='utf-8') as f:
                fbas = FBAS.from_json(json.load(f))
            logging.info("Finding Erdos-Renyi parameter for network at %s" % file_path)
        else:
            n_orgs = int(args['--n_orgs'])
            fbas = gen_symmetric_fbas(n_orgs, output=None)
            logging.info("Finding Erdos-Renyi parameter for symmetric network with %d organizations" % n_orgs)
        p = find_p(fbas)
        logging.info("p = %f" % p)
        # print the corresponding average degree:
        logging.info("Average degree: %f" % (len(fbas.validators())*p))

    elif args['--check-assumptions']:
        logging.info("Checking failure assumptions")
        file_path = os.path.expanduser(args['--fbas'].strip())
        with open(file_path, 'r', encoding='utf-8') as f:
            fbas = FBAS.from_json(json.load(f))
        g = nx.Graph()
        for v in fbas.validators():
            g.add_node(v)
            for w in fbas.metadata[v]['peers']:
                g.add_edge(v, w)
        if check_failure_assumptions(g, fbas):
            logging.info("Failure assumptions are satisfied")
        else:
            logging.info("Failure assumptions are not satisfied")