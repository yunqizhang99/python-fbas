from copy import copy
from typing import Any
import networkx as nx

def max_simple_path(graph:nx.DiGraph):
    """
    Returns the maximal length of simple paths in the graph (the graph may have cycles or be
    disconnected).

    We just try all lengths l, starting with 1, until no more nodes can be reached. That's not very
    smart but it should work for small-enough graphs.
    """
    l = 1
    reachable:dict[Any, Any] = {}
    last_reachable:dict[Any, Any] = {}
    while True:
        inc = False
        for n in graph.nodes:
            # compute the set of nodes that can be reached from n in at most l steps
            reachable[n] = set(nx.single_source_shortest_path_length(graph, n, cutoff=l).keys())
            if len(reachable[n]) < len(graph.nodes):
                inc = True
                break
        if reachable == last_reachable:
            return l-1
        last_reachable = copy(reachable)
        if inc:
            l += 1
        else:
            return l
    