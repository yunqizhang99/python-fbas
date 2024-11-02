"""
Federated Byzantine Agreement System (FBAS) represented as graphs.
"""

from typing import Any, Optional, Tuple
from collections.abc import Collection
import logging
from dataclasses import dataclass
from pprint import pformat
import networkx as nx
from python_fbas.utils import powerset

# TODO we don't really need QSet, do we?
# TODO use nx?

@dataclass
class FBASGraphNode:
    label: Any
    threshold: int = 0
    children: Optional[set['FBASGraphNode']] = None
    meta: Optional[dict] = None

    def __post_init__(self):
        if self.children is None:
            self.children = set()
        if self.meta is None:
            self.meta = dict()
        if (self.threshold < 0
            or self.threshold > len(self.children)
            or (self.threshold == 0 and self.children)):
            raise ValueError(f"FBASGraphNode failed validation: {self}")
        
    def __eq__(self, other):
        return isinstance(other, FBASGraphNode) and self.label == other.label
    
    def __hash__(self):
        return hash(self.label)
    
    def __str__(self):
        def pretty_id(i):
            return hash(i) if isinstance(i, QSet) else i
        return f"FBASGraphNode({pretty_id(self.label)}, {self.threshold}, {set(pretty_id(c.label) for c in self.children)})"

def freeze_qset(qset: dict) -> Tuple[int, frozenset]:
    """
    Expects a JSON-serializable quorum-set (in stellarbeat.io format) and returns a hashable version for use in collections.
    """
    threshold = int(qset['threshold'])
    members = frozenset(qset['validators']) | frozenset(freeze_qset(iq) for iq in qset['innerQuorumSets'])
    assert 0 <= threshold <= len(members)
    assert not (threshold == 0 and len(members) > 0)
    return (threshold, members)

class FBASGraph:
    """
    A graph whose nodes are either validators or QSets.
    Each node has otional attributes: a optional treshold and metadata attibutes
    """
    graph: nx.DiGraph
    validators: set # only a subset of the nodes in the graph represent validators

    def __init__(self, graph=None, validators=None):
        if graph is None:
            self.graph = nx.DiGraph()
        else:
            self.graph = graph
        if validators is None:
            self.validators = set()
        else:
            self.validators = validators


    def check_integrity(self):
        """Basic integrity checks"""
        if not self.validators <= self.graph.nodes():
            raise ValueError(f"Some validators are not in the graph: {self.validators - self.graph.nodes()}")
        for n, attrs in self.graph.nodes(data=True):
            if 'threshold' not in attrs:
                assert self.graph.out_degree(n) == 1
            else:
                if attrs['threshold'] < 0 or attrs['threshold'] > len(self.graph.out_degree(n)):
                    raise ValueError(f"Integrity check failed: threshold of {n} not in [0, out_degree={self.graph.out_degree(n)}]")
        for v in self.validators:
            if self.graph.out_degree(v) != 1:
                raise ValueError(f"Integrity check failed: validator {v} has no qset")
        # for v in self.graph.nodes():
        #     if n not in self.validators and self.graph.out_degree(n) == 1:
        #         raise ValueError(f"Integrity check failed:  qset {n} has a single successor")
                
    def stats(self):
        """Compute some basic statistics"""
        def thresholds_distribution():
            return {t: sum(1 for _, attrs in self.graph.nodes(data=True) if attrs['threshold'] == t)
                    for t in self.graph.get_node_attributes('threshold').values()}
        return {
            'num_edges' : len(self.graph.edges()),
            'thresholds_distribution' : thresholds_distribution()
        }
        
    def add_validator(self, v:Any) -> None:
        """Add a validator to the graph"""
        self.graph.add_node(v)
        self.validators.add(v)

    def update_validator(self, v: Any, qset: Optional[dict] = None, attrs: Optional[dict] = None) -> None:
        """
        Add the validator v to the graph if it does not exist.
        Otherwise:
            - Update its attributes with attrs (existing attributes not in attrs remain unchanged).
            - Replace its outgoing edge with an edge to the given qset.
        Expects a qset, if given, in JSON-serializable stellarbeat.io format.
        """
        if attrs:
            self.graph.add_node(v, **attrs)
        else:
            self.graph.add_node(v)
        if qset:
            fqs = self.add_qset(qset)
            self.graph.remove_edges_from(self.graph.out_edges(v))
            self.graph.add_edge(v, fqs)
        self.validators.add(v)
    
    def add_qset(self, qset: dict) -> Tuple[int, frozenset]:
        """
        Takes a qset as a JSON-serializable dict in stellarbeat.io format.
        Returns the qset if it already exists, otherwise adds it to the graph.
        """
        for iq in qset['innerQuorumSets']:
            self.add_qset(iq)
        vs = qset['validators']
        for v in vs:
            self.add_validator(v)
        n = freeze_qset(qset)
        self.graph.add_node(n, threshold=n[0])
        for member in n[1]:
            self.graph.add_edge(n, member)
        return n

    def __str__(self):
        # number qset nodes from 1 to n:
        qset_nodes = {n for n in self.graph.nodes() if not n in self.validators}
        qset_index = {n:i for i,n in enumerate(qset_nodes, 1)}
        def node_repr(n):
            if n in self.validators:
                return f"{n}"
            else:
                return f"q{qset_index[n]}"
        res = {node_repr(n) : f"({t}, {map(node_repr,self.graph.successors(n))})"
                for n,t in self.graph.nodes('threshold')}
        return pformat(res)

    @staticmethod
    def from_json(data : list, from_stellarbeat = False) -> 'FBASGraph':
        """
        Create a FBASGraph from a list of validators in serialized stellarbeat.io format.
        """
        # first do some validation
        validators = []
        keys = set()
        for v in data:
            if not isinstance(v, dict):
                logging.warning("Ignoring non-dict entry: %s", v)
                continue
            if 'publicKey' not in v:
                logging.warning(
                    "Entry is missing publicKey, skipping: %s", v)
                continue
            if (from_stellarbeat and (
                    ('isValidator' not in v or not v['isValidator'])
                    or ('isValidating' not in v or not v['isValidating']))):
                logging.warning(
                    "Ignoring non-validating validator: %s (name: %s)", v['publicKey'], v.get('name'))
                continue
            if 'quorumSet' not in v or v['quorumSet'] is None:
                logging.warning(
                    "Using empty QSet for validator missing quorumSet: %s", v['publicKey'])
                v['quorumSet'] = {'threshold': 0,
                                  'validators': [], 'innerQuorumSets': []}
            if v['publicKey'] in keys:
                logging.warning(
                    "Ignoring duplicate validator: %s", v['publicKey'])
                continue
            keys.add(v['publicKey'])
            validators.append(v)
        # now create the graph:
        fbas = FBASGraph()
        for v in validators:
            fbas.update_validator(v['publicKey'], v['quorumSet'], v)
        return fbas

    def flatten_diamonds(self) -> None:
        """
        Roughly speaking, we identify all the "diamonds" in the graph and "flatten" them.
        This operation preserves quorum intersection.
        """
        pass
        # def collapse_diamond(n: FBASGraphNode) -> bool:
        #     """collapse diamonds with > 1/2 threshold"""
        #     if not len(n.children) > 1:
        #         return False
        #     child = next(iter(n.children))
        #     if not child.children:
        #         return False
        #     grandchild = next(iter(child.children))
        #     is_diamond = (
        #         all(self.parents[c] == {n} for c in n.children)
        #         and all(c.children == set([grandchild]) for c in n.children)
        #         and 2*n.threshold > len(n.children))
        #     if is_diamond:
        #         logging.debug("Collapsing diamond at: %s", n)
        #         assert n not in self.validators
        #         self.validators |= set([n])
        #         for c in n.children:
        #             del self.parents[c]
        #         if n != grandchild:
        #             n.children = set([grandchild])
        #             n.threshold = 1
        #             self.parents[grandchild] |= set([n])
        #         else:
        #             n.children = set()
        #             n.threshold = 0
        #     return is_diamond

        # # now collapse nodes until nothing changes:
        # while True:
        #     self.check_integrity() # for debugging
        #     for n in self.nodes:
        #         if collapse_diamond(n):
        #             break
        #     else:
        #         return

    def threshold(self, n: Any) -> int:
        """
        Returns the threshold of the given node.
        """
        if 'threshold' in self.graph.nodes[n]:
            return self.graph.nodes[n]['threshold']
        else:
            assert self.graph.out_degree(n) == 1 # canary
            return 1
    
    def is_sat(self, n: Any, s: Collection) -> bool:
        """
        Returns True if and only if n's agreement requirements are satisfied by s.
        """
        assert s <= self.validators
        if all(c in self.validators for c in self.graph.successors(n)):
            assert n not in self.validators and 'threshold' in self.graph.nodes[n] # canary
            return self.threshold(n) <= sum(1 for c in self.graph.successors(n) if c in s)
        else:
            return self.threshold(n) <= sum(1 for c in self.graph.successors(n) if self.is_sat(c , s))
        
    def qset_nodes(self, n: Any) -> frozenset:
        """
        If n is a qset node, returns the set of graph nodes that form the full qset below n and including n.
        Otherwise just return {n}.
        """
        assert n in self.graph.nodes
        if n in self.validators:
            return frozenset([n])
        else:
            return frozenset([n]) | frozenset.union(*[self.qset_nodes(c) for c in self.graph.successors(n)])
    
    def qset_node_of(self, n: Any) -> tuple[int, frozenset]:
        """
        Returns the qset node of the given validator node (i.e. its successor).
        """
        assert n in self.validators
        assert self.graph.out_degree(n) == 1
        return next(self.graph.successors(n))

    def is_quorum(self, vs: Collection) -> bool:
        """
        Returns True if and only if s is a non-empty quorum.
        Not efficient.
        """
        if not vs:
            return False
        assert vs <= self.validators
        return all(self.is_sat(v, vs) for v in vs)
            
    def find_disjoint_quorums(self) -> Optional[tuple[set, set]]:
        """
        Naive, brute-force search for disjoint quorums.
        Warning: use only for very small fbas graphs.
        """
        assert len(self.validators) < 10
        quorums = [q for q in powerset(list(self.validators)) if self.is_quorum(q)]
        return next(((q1, q2) for q1 in quorums for q2 in quorums if not (q1 & q2)), None)