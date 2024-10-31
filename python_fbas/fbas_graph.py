from typing import Any, Optional
import logging
from dataclasses import dataclass
from pprint import pformat
from python_fbas.fbas import QSet


# graph representation of an FBAS:
# TODO: The whole approach seems wrong. We need to differentiate between validators and qsets! We cannot collapse validators!
# Or maybe it's fine if all qsets self-intersect? But what if they don't?
# Maybe an easy way to fix it is to only collapse stuff that has more than one child (since that can't be validators).
# But in the end when checking intersection we still need to check that there are _validators_ in common, not just qsets.

@dataclass
class FBASGraphNode:
    id: Any # equality based on __eq__
    threshold: int = 0
    children: Optional[set['FBASGraphNode']] = None

    def __post_init__(self):
        if self.children is None:
            self.children = set()
        if (self.threshold < 0
            or self.threshold > len(self.children)
            or (self.threshold == 0 and self.children)):
            raise ValueError(f"FBASGraphNode failed validation: {self}")
        
    def __eq__(self, other):
        return self.id == other.id
    
    def __hash__(self):
        return hash(self.id)
    
    def __str__(self):
        def pretty_id(i):
            return hash(i) if isinstance(i, QSet) else i
        return f"FBASGraphNode({pretty_id(self.id)}, {self.threshold}, {set(pretty_id(c.id) for c in self.children)})"

class FBASGraph:
    nodes: set[FBASGraphNode]
    validators: set[FBASGraphNode]
    parents: dict[FBASGraphNode, set[FBASGraphNode]]

    def __init__(self):
        self.nodes = set()
        self.validators = set()
        self.parents = dict()

    def check_integrity(self):
        for v in self.validators:
            if v not in self.nodes:
                raise ValueError(f"Integrity check failed: validator {v.id} not in nodes")
        for n in self.nodes:
            if n.threshold < 0 or n.threshold > len(n.children):
                raise ValueError(f"Integrity check failed: threshold of {n.id} not in [0, len(children)]")
            for c in n.children:
                if c not in self.nodes:
                    raise ValueError("Integrity check failed: a child is not in the set of nodes")
                if n not in self.parents[c]:
                    raise ValueError("Integrity check failed: parent is not in the parents set")
        for n,ps in self.parents.items():
            for p in ps:
                if p not in self.nodes:
                    raise ValueError("Integrity check failed: a parent is not in the set of nodes")
                if n not in p.children:
                    raise ValueError("Integrity check failed: child is not in the children set")
                
    def stats(self):
        def thresholds_distribution():
            return {n.threshold : sum(1 for m in self.nodes if m.threshold == n.threshold) for n in self.nodes}
        return {
            'num_edges' : sum(len(ps) for ps in self.parents.values()),
            'thresholds_distribution' : thresholds_distribution()
        }

    def add_node(self, id: Any) -> FBASGraphNode:
        for n in self.nodes:
            if n.id == id:
                return n
        n = FBASGraphNode(id)
        self.nodes.add(n)
        return n
        
    def add_validator(self, v: Any) -> FBASGraphNode:
        for w in self.validators:
            if w.id == v:
                return w
        assert v not in self.nodes
        n = self.add_node(v)
        self.validators.add(n)
        return n

    def add_qset(self, qset: QSet) -> FBASGraphNode:
        for n in self.nodes:
            if n.id == qset:
                return n
        iqs = [self.add_qset(iq) for iq in qset.inner_qsets]
        vs = [self.add_validator(v) for v in qset.validators]
        children = set(iqs) | set(vs)
        n = FBASGraphNode(qset, qset.threshold, children)
        self.nodes.add(n)
        for c in children:
            self.parents[c] = self.parents.get(c, set()) | {n}
        return n
    
    def update_validator(self, v, qset : QSet) -> FBASGraphNode:
        qset_node = self.add_qset(qset)
        n = self.add_validator(v)
        n.threshold = 1
        n.children = {qset_node}
        self.parents[qset_node] = self.parents.get(qset_node, set()) | {n}
        return n

    def __str__(self):
        def pretty_id(n):
            return hash(n.id) if isinstance(n.id, QSet) else n.id
        return pformat({pretty_id(n): (n.threshold, [pretty_id(c) for c in n.children]) for n in self.nodes}, indent=2)

    @staticmethod
    def from_json(data : list, from_stellarbeat = False) -> 'FBASGraph':
        # first sanitize the data a bit:
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
        graph = FBASGraph()
        for v in validators:
            qset = QSet.from_json(v['quorumSet'])
            graph.update_validator(v['publicKey'], qset)
        return graph

    def collapse(self) -> None:
        """Make the FBASGraph smaller by repeatedly collpasing nodes"""
        
        def collapse_diamond(n: FBASGraphNode) -> bool:
            """collapse diamonds with > 1/2 threshold"""
            if not len(n.children) > 1:
                return False
            child = next(iter(n.children))
            if not child.children:
                return False
            grandchild = next(iter(child.children))
            is_diamond = (
                all(self.parents[c] == {n} for c in n.children)
                and all(c.children == set([grandchild]) for c in n.children)
                and 2*n.threshold > len(n.children))
            if is_diamond:
                logging.debug("Collapsing diamond at: %s", n)
                for c in n.children:
                    del self.parents[c]
                if n != grandchild:
                    n.children = set([grandchild])
                    n.threshold = 1
                    self.parents[grandchild] |= set([n])
                else:
                    n.children = set()
                    n.threshold = 0
            return is_diamond

        # now collapse nodes until nothing changes:
        while True:
            self.check_integrity()
            for n in self.nodes:
                if collapse_diamond(n):
                    break
            else:
                return