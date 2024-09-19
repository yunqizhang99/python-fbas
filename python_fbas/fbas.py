"""
A module for representing and working with Federated Byzantine Agreement Systems (FBAS).
"""
# TODO debug prints

from dataclasses import dataclass
import logging
from itertools import combinations
from typing import Any, Optional
import networkx as nx
from .utils import fixpoint


@dataclass(frozen=True)
class QSet:

    """Stellar's so-called quorum sets (which are NOT sets of quorums, but instead represent sets of quorum slices)"""

    threshold: int
    validators: frozenset
    inner_qsets: frozenset  # a set of QSets
    metadata: Optional[dict]

    def __post_init__(self):
        # check that no validator appears twice in the qset:
        def unique_validators(qs: QSet):
            return (all(not (iqs1.all_validators() & iqs2.all_validators()) for iqs1 in qs.inner_qsets for iqs2 in qs.inner_qsets if iqs1 != iqs2)
                    and (all(not (qs.validators & iqs.all_validators()) for iqs in self.inner_qsets))
                    and (unique_validators(iqs) for iqs in self.inner_qsets))
        # make some validity checks:
        valid = (self.threshold >= 0
                 and self.threshold <= len(self.validators) + len(self.inner_qsets)
                 and all(isinstance(qs, QSet) for qs in self.inner_qsets)
                 and unique_validators(self))
        if not valid:
            raise ValueError(f"QSet failed validation: {self}")

    def __bool__(self):
        return bool(self.validators | self.inner_qsets)

    def __str__(self):
        return f"QSet({self.threshold},{str(set(self.validators)) if self.validators else '{}'},{str(set(self.inner_qsets)) if self.inner_qsets else '{}'})"

    @staticmethod
    def make(threshold: int, validators, inner_qsets, metadata: Optional[dict] = None):
        return QSet(threshold, frozenset(validators), frozenset(inner_qsets), metadata)

    @staticmethod
    def from_json(data: dict):
        match data:
            case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqss}:
                inner_qsets = [QSet.from_json(qs) for qs in iqss]
                return QSet.make(t, vs, inner_qsets)
            case _: raise ValueError(f"Invalid QSet JSON: {data}")

    def elements(self):
        return self.validators | self.inner_qsets

    def depth(self):
        return 1 + (0 if not self.inner_qsets else max(iqs.depth() for iqs in self.inner_qsets))

    def slices(self):
        return combinations(self.elements(), self.threshold)

    def blocking_threshold(self):
        return len(self.elements()) - self.threshold + 1

    def v_blocking_sets(self):
        return combinations(self.elements(), self.blocking_threshold())

    def sat(self, validators):
        """Whether the agreement requirements encoded by this QSet are satisfied by the given set of validators."""
        sat_inner_qsets = {
            iqs for iqs in self.inner_qsets if iqs.sat(validators)}
        return len((self.validators & set(validators)) | sat_inner_qsets) >= self.threshold

    def all_qsets(self):
        """Returns the set containing self and all inner QSets appearing (recursively) in this QSet."""
        return frozenset((self,)) | self.inner_qsets | frozenset().union(*(iqs.all_qsets() for iqs in self.inner_qsets))

    def blocked(self, validators):
        """
        Whether this QSet is blocked by the given set of validators (meaning all slices include a member of the set validators).
        """
        def directly_blocked(qs, xs):
            return len(qs.elements() & xs) > len(qs.elements()) - qs.threshold

        def _blocked(xs):
            return xs | {qs for qs in self.all_qsets() - xs if directly_blocked(qs, xs)}
        return self in fixpoint(_blocked, frozenset(validators))

    def all_validators(self):
        """Returns the set of all validators appearing (recursively) in this QSet."""
        return self.validators | set().union(*(iqs.all_validators() for iqs in self.inner_qsets))


def qset_intersection_bound(qset1, qset2):
    """
    Returns a lower bound on the number of validators in common in any two slices of qset1 and qset2.
    Assumes that the qsets have at most 2 levels.
    Additionally, if qset1.validators and qset2.validators are empty and their inner qsets represent organizations,
    this is also a lower bound on the number of organizations one must compromise to obtain two slices that have only compromised
    organizations in common.
    """
    if qset1.depth() > 2 or qset2.depth() > 2:
        raise ValueError(
            "direct_safety_margin only works for qsets with at most 2 levels")
    #  in order to guarantee intersection, we require common inner-qsets to have a threshold of more than half:
    if any(2*qs.threshold < len(qs.elements()) for qs in qset1.inner_qsets | qset2.inner_qsets):
        return 0
    n1, n2 = len(qset1.elements()), len(qset2.elements())
    t1, t2 = qset1.threshold, qset2.threshold
    common = qset1.elements() & qset2.elements()
    nc = len(common)
    # worst-case number of common elements in a slice of qset1
    m1 = (t1 + nc) - n1
    # worst-case number of common elements in a slice of qset2
    m2 = (t2 + nc) - n2
    # We could make it more precise by tracking the size of org intersections (e.g. the minimal intersection is 2 for a 3/4 org)
    return max((m1 + m2) - nc, 0)


@dataclass
class FBAS:
    qset_map: dict[Any, QSet]
    metadata: Optional[dict] = None

    def __post_init__(self):
        if not all(isinstance(v, QSet) for v in self.qset_map.values()):
            bad_qset = next(v for v in self.qset_map.values()
                            if not isinstance(v, QSet))
            raise ValueError(
                f"FBAS failed validation: the inner QSet {bad_qset} is not a QSet")
        missing = []
        for qs in self.all_qsets():
            for v in qs.validators:
                if v not in self.validators():
                    logging.warning(
                        "Validator %s appears in QSet %s but not in the FBAS's key range, "
                        "adding with empty qset",
                        v,
                        qs
                    )
                    missing.append(v)
        if missing:
            self.qset_map.update({v: QSet.make(0, [], []) for v in missing})

    def sanitize(self) -> 'FBAS':
        # A validator is invalid if it has 'isValidator' or 'isValidating' set to False or its QSet contains a validator that does not appear in the qset_map:
        def is_valid(v, qsm):
            return (
                v in self.metadata
                and self.metadata[v].get('isValidator')
                and self.metadata[v].get('isValidating')
                and qsm[v].threshold > 0
                and all(w in qsm for w in qsm[v].all_validators())
            )
        # remove all invalid validators from the qset_map until nothing changes:
        new_qset_map = self.qset_map
        while True:
            valid_validators = {v for v in new_qset_map if is_valid(v, new_qset_map)}
            if valid_validators == new_qset_map.keys():
                break
            new_qset_map = {v: qs for v,
                            qs in new_qset_map.items() if v in valid_validators}

        return FBAS(new_qset_map, self.metadata)

    @staticmethod
    def from_json(data: list):
        if not isinstance(data, list):
            raise ValueError("json_data must be a list")
        # first we filter out or fix invalid entries:
        valid_data = []
        seen_keys = set()
        for v in data:
            if 'publicKey' not in v:
                logging.warning(
                    "Entry is missing publicKey, skipping: %s", v)
                continue
            if 'publicKey' not in v or 'quorumSet' not in v or v['quorumSet'] is None:
                logging.warning(
                    "Using empty QSet for validator missing quorumSet: %s", v['publicKey'])
                v['quorumSet'] = {'threshold': 0,
                                  'validators': [], 'innerQuorumSets': []}
            if v['publicKey'] in seen_keys:
                logging.warning(
                    "Ignoring duplicate validator: %s", v['publicKey'])
                continue
            seen_keys.add(v['publicKey'])
            valid_data.append(v)
        # now we create the qset map:
        qsm = {v['publicKey']: QSet.from_json(
            v['quorumSet']) for v in valid_data}
        # finally we dump all remaining keys into the metadata:
        meta = {
            v['publicKey']: {
                k: val for k, val in v.items() if k not in {'publicKey', 'quorumSet'}
            }
            for v in data
        }
        return FBAS(qsm, meta)

    def is_org_structured(self) -> bool:
        """
        An FBAS is org-structured when, for each depth-1 QSet qs appearing in the FBAS,
        the validators of qs do not appear in any other QSet.
        """
        depth_1_qsets = {qs for qs in self.all_qsets() if qs.depth() == 1}
        return all(all(not (qs1.validators & qs2.validators) for qs2 in self.all_qsets() if qs2 != qs1) for qs1 in depth_1_qsets)

    def org_of_qset(self, qs: QSet) -> Optional[str]:
        """
        If this qset represents an organization, i.e. all validators have the same homeDomain, return
        that domain. Otherwise return None.
        """
        home_domains = {
            self.metadata[v].get('homeDomain')
            for v in qs.validators if v in self.metadata and 'homeDomain' in self.metadata[v]
        }
        if len(home_domains) == 1:
            return home_domains.pop()
        return None

    def is_quorum(self, validators) -> bool:
        """
        A set of validators is a quorum if it satisfies the agreement requirements of all its members.
        Throws an error if a validator is not in the FBAS.
        """
        return all(qs.sat(validators) for qs in (self.qset_map[v] for v in validators))

    def to_graph(self):
        """
        Returns a directed graph where the nodes are validators and there is an edge from v to w if
        v appears anywhere is in w's QSet.
        """
        g = nx.DiGraph()
        for v, qs in self.qset_map.items():
            for w in qs.all_validators():
                g.add_edge(v, w)
        return g

    def to_mixed_graph(self):
        """
        Returns a directed graph where the nodes are validators and QSets. There is an edge from
        each validator to its QSet, and an edge from each QSet to each of its direct members.
        """
        # first collect all the qsets appearing anywhere
        qsets = frozenset().union(*(qs.all_qsets() for qs in self.qset_map.values()))
        # create a graph with a node for each validator and each qset
        g = nx.DiGraph()
        # add an edge from each validator to its QSet:
        g.add_edges_from(list(self.qset_map.items()))
        # add an edge from each QSet to each of its members:
        for qs in qsets:
            for x in qs.validators | qs.inner_qsets:
                g.add_edge(qs, x)
        return g

    def to_weighed_graph(self):
        def weights(qs: QSet):
            w = 1/len(qs.elements())
            return {v: w for v in qs.validators} | \
                {v: w*weights(qs)[v]
                 for qs in qs.inner_qsets for v in weights(qs).keys()}
        g = nx.DiGraph()
        for v, qs in self.qset_map.items():
            for w, weight in weights(qs).items():
                g.add_edge(v, w, weight=weight)
        return g

    def closure(self, S):
        """Computes the closure of the set of validators S"""
        def _blocked(xs):
            return {v for v, qs in self.qset_map.items() if qs.blocked(xs)} | xs
        return fixpoint(_blocked, set(S))

    def validators(self):
        """Returns the set of keys of self.qset_map"""
        return self.qset_map.keys()

    def all_qsets(self):
        """Returns the set containing all QSets appearing in this FBAS."""
        return frozenset().union(*(qs.all_qsets() for qs in self.qset_map.values()))

    def elements(self):
        return frozenset(self.validators() | self.all_qsets())

    def max_scc(self):
        # TODO: if there's more than one, use pagerank to find the "right" one?
        return max(nx.strongly_connected_components(self.to_graph()), key=len)

    def min_scc_intersection_bound(self):
        max_scc_qsets = {self.qset_map[v] for v in self.max_scc()}
        return min(qset_intersection_bound(q1, q2) for q1 in max_scc_qsets for q2 in max_scc_qsets)

    def intersection_check_heuristic(self):
        """
        Compute a maximal directly-intertwined set (i.e. max clique) in the max scc, then check its closure is the whole fbas.
        If not, repeat with another maximal clique.
        Do that a number of times.
        Also fail immediately if the undirected version of the graph is not connected.
        """
        pass

    def splitting_set_bound_heuristic(self):
        pass

    # TODO: would be nice to use org names as new validator name
    def collapse_qsets(self, new_name=None):
        """
        A QSet is collapsible if it can safely be replaced by a single (new) validator without impacting quorum intersection.
        This method uses a heuristic to identify collapsible QSets and returns a new fbas where all identified QSets have been replaced by a new validator.
        It might then be easier to check quorum intersection.
        """
        def is_collapsible(qset):
            res = (
                # no inner QSets:
                not qset.inner_qsets
                # the validators of this QSet do not appear anywhere else:
                and all(not (qset.validators & qs.validators) for qs in self.all_qsets() if qs != qset)
                # the validators of this QSet all have the same QSet:
                and len({self.qset_map[v] for v in qset.validators}) == 1
                # threshold is greater than half:
                and 2*qset.threshold > len(qset.validators))
            if new_name:
                print(f"qset {new_name(qset)} collapsible: {res}")
            return res
        # for each collapsible QSet, create a new validator:
        collapsible = list(
            {qs for qs in self.all_qsets() if is_collapsible(qs)})
        # TODO why is this not a function?
        validator_of_qset = {
            qs: new_name(qs)
            if new_name and new_name(qs) else collapsible.index(qs) for qs in collapsible
        }
        if validator_of_qset.values() & self.validators():
            raise Exception(
                "New validators clash with existing validators (should not happen if validators are identified by strings)")
        # now replace all the collapsible qsets:

        def replace_collapsible(qs):
            if qs in collapsible:
                return QSet.make(1, [validator_of_qset[qs]], [])
            else:
                return QSet.make(
                    qs.threshold,
                    qs.validators | {validator_of_qset[cqs]
                                     for cqs in qs.inner_qsets & set(collapsible)},
                    (ncqs for ncqs in qs.inner_qsets if ncqs not in collapsible))

        def qset_of_collapsible(qset):
            qset_of_members = self.qset_map[next(
                iter(qset.validators))] if qset.validators else None
            return replace_collapsible(qset_of_members) if qset_of_members not in collapsible else QSet.make(1, [validator_of_qset[qset_of_members]], [])
        # TODO we also have to remove the validators that are no longer needed!
        new_qset_map = ({v: replace_collapsible(qs) for v, qs in self.qset_map.items()} |
                        {validator_of_qset[qs]: qset_of_collapsible(qs) for qs in collapsible})
        return FBAS(new_qset_map)
