from utils import fixpoint
from dataclasses import dataclass
import networkx as nx

@dataclass(frozen=True)
class QSet:
    threshold: int
    validators: frozenset
    inner_qsets: frozenset

    def __post_init__(self):
        valid = (self.threshold > 0
            and self.threshold <= len(self.validators) + len(self.inner_qsets)
            and all(isinstance(qs, QSet) for qs in self.inner_qsets))
        if not valid:
            raise Exception(f"QSet failed validation: {self}")

    def __bool__(self):
        return bool(self.validators | self.inner_qsets)

    def __str__(self):
        return f"QSet({self.threshold},{str(set(self.validators)) if self.validators else '{}'},{str(set(self.inner_qsets)) if self.inner_qsets else '{}'})"

    @staticmethod
    def make(threshold, validators, inner_qsets):
        return QSet(threshold, frozenset(validators), frozenset(inner_qsets))

    def sat(self, validators):
        """Whether the agreement requirements encoded by this QSet are satisfied by the given set of validators."""
        sat_inner_qsets = {iqs for iqs in self.inner_qsets if iqs.sat(validators)}
        return len((self.validators & set(validators)) | sat_inner_qsets) >= self.threshold

    def all_qsets(self):
        """Returns the set containing self and all inner QSets appearing (recursively) in this QSet."""
        return frozenset((self,)) | self.inner_qsets | frozenset().union(*(iqs.all_qsets() for iqs in self.inner_qsets))

    def blocked(self, validators):
        """
        Whether this QSet is blocked by the given set of validators (meaning all slices include a member of the set validators).
        """
        def directly_blocked(q, xs):
            members = (q.validators | q.inner_qsets)
            return len(members & xs) > len(members) - q.threshold
        def _blocked(xs):
            return xs | {q for q in self.all_qsets() if directly_blocked(q, xs)}
        return self in fixpoint(_blocked, frozenset(validators))

    def all_validators(self):
        """Returns the set of all validators appearing (recursively) in this QSet."""
        return self.validators | set().union(*(iqs.all_validators() for iqs in self.inner_qsets))
    
    @staticmethod
    def from_stellarbeat_json(data : dict):
        return QSet.make(data['threshold'], data['validators'],
                         [QSet.from_stellarbeat_json(qs) for qs in data['innerQuorumSets']])

@dataclass
class FBAS:
    qset_map: dict

    def __post_init__(self):
        if not all(isinstance(v, QSet) for v in self.all_qsets()):
            raise Exception(f"FBAS failed validation: an inner QSet is not a QSet")
        for qs in self.all_qsets():
            for v in qs.validators:
                if v not in self.qset_map.keys():
                    raise Exception(f"Validator {v} appears in QSet {qs} but not in the FBAS's key range")

    @staticmethod
    def from_stellarbeat_json(data : list):
        return FBAS({v['publicKey'] : QSet.from_stellarbeat_json(v['quorumSet']) for v in data})

    def is_quorum(self, validators):
        """
        A set of validators is a quorum if it satisfies the agreement requirements of all its members.
        Throws an error if a validator is not in the FBAS.
        """
        return all(qs.sat(validators) for qs in (self.qset_map[v] for v in validators))

    def to_graph(self):
        # first collect all the qsets appearing anywhere
        qsets = frozenset().union(*(qs.all_qsets() for qs in self.qset_map.values()))
        # create a graph with a node for each validator and each qset
        g = nx.DiGraph()
        g.add_edges_from(list(self.qset_map.items()))
        for qs in qsets:
            for x in qs.validators | qs.inner_qsets:
                g.add_edge(qs, x)
        return g

    def closure(self, S):
        """Computes the closure of the set of validators S"""
        def _blocked(xs):
            return {v for v, qs in self.qset_map.items() if qs.blocked(xs)} | xs
        return fixpoint(_blocked, set(S))

    def all_qsets(self):
        """Returns the set containing all QSets appearing in this FBAS."""
        return frozenset().union(*(qs.all_qsets() for qs in self.qset_map.values()))