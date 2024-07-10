from utils import fixpoint
from dataclasses import dataclass
import networkx as nx

@dataclass(frozen=True)
class QSet:
    threshold: int
    validators: frozenset
    innerQSets: frozenset

    def __bool__(self):
        return bool(self.validators | self.innerQSets)

    def __str__(self):
        return f"QSet({self.threshold},{str(set(self.validators)) if self.validators else '{}'},{str(set(self.innerQSets)) if self.innerQSets else '{}'})"

    @staticmethod
    def make(threshold, validators, innerQSets):
        return QSet(threshold, frozenset(validators), frozenset(innerQSets))

    def sat(self, validators):
        """Whether the agreement requirements encoded by this QSet are satisfied by the given set of validators."""
        sat_innerQSets = {iqs for iqs in self.innerQSets if iqs.sat(validators)}
        return len((self.validators & set(validators)) | sat_innerQSets) >= self.threshold

    def all_QSets(self):
        """Returns the set containing self and all innerQSets appearing (recursively) in this QSet."""
        return frozenset((self,)) | self.innerQSets | frozenset().union(*(iqs.all_QSets() for iqs in self.innerQSets))

    def blocked(self, validators):
        """
        Whether this QSet is blocked by the given set of validators (meaning all slices include a member of the set validators).
        """
        def directly_blocked(q, xs):
            members = (q.validators | q.innerQSets)
            return len(members & xs) > len(members) - q.threshold
        def _blocked(xs):
            return xs | {q for q in self.all_QSets() if directly_blocked(q, xs)}
        return self in fixpoint(_blocked, frozenset(validators))

    def all_validators(self):
        """Returns the set of all validators appearing (recursively) in this QSet."""
        return self.validators | set().union(*(iqs.all_validators() for iqs in self.innerQSets))

@dataclass
class FBAS:
    qset_map: dict

    def is_quorum(self, validators):
        """
        A set of validators is a quorum if it satisfies the agreement requirements of all its members.
        Throws an error if a validator is not in the FBAS.
        """
        return all(qs.sat(validators) for qs in (self.qset_map[v] for v in validators))

    def to_graph(self):
        # first collect all the qsets appearing anywhere
        qsets = frozenset().union(*(qs.all_QSets() for qs in self.qset_map.values())) \
            | frozenset(self.qset_map.values())
        # create a graph with a node for each validator and each qset
        G = nx.Graph()
        G.add_edges_from(list(self.qset_map.items()))
        for qs in qsets:
            for iqs in qs.innerQSets:
                G.add_edge(qs, iqs)
        return G

    def closure(self, S):
        """Computes the closure of the set of validators S"""
        def _blocked(xs):
            return {v for v in self.qset_map.keys() if self.qset_map[v].blocked(xs)} | xs
        return fixpoint(_blocked, set(S))
