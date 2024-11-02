"""
A module for representing and working with Federated Byzantine Agreement Systems (FBAS).
"""

from dataclasses import dataclass
import logging
from pprint import pformat
from functools import lru_cache
from itertools import combinations, product, islice
from typing import Any, Optional, Literal
import networkx as nx
from python_fbas.utils import fixpoint


@dataclass(frozen=True)
class QSet:

    # TODO: might make sense to make this a singleton

    """Stellar's so-called quorum sets (which are NOT sets of quorums, but instead represent sets of quorum slices)"""

    threshold: int
    validators: frozenset
    inner_qsets: frozenset  # a set of QSets

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
        return f"""QSet(
            {self.threshold},
            {str(set(self.validators)) if self.validators else '{}'},
            {str(set(self.inner_qsets)) if self.inner_qsets else '{}'})"""

    @staticmethod
    def make(threshold: int, validators, inner_qsets):
        return QSet(threshold, frozenset(validators), frozenset(inner_qsets))

    @staticmethod
    def from_json(data: dict):
        match data:
            case {'threshold': t, 'validators': vs, 'innerQuorumSets': iqss}:
                inner_qsets = [QSet.from_json(qs) for qs in iqss]
                return QSet.make(t, vs, inner_qsets)
            case _: raise ValueError(f"Invalid QSet JSON: {data}")

    def to_json(self):
        return {
            'threshold': self.threshold,
            'validators': list(self.validators),
            'innerQuorumSets': [iqs.to_json() for iqs in self.inner_qsets]
        }

    def elements(self):
        return self.validators | self.inner_qsets

    def depth(self):
        return 1 + (0 if not self.inner_qsets else max(iqs.depth() for iqs in self.inner_qsets))

    def level_1_slices(self):
        return combinations(self.elements(), self.threshold)

    def blocking_threshold(self):
        return len(self.elements()) - self.threshold + 1

    def level_1_v_blocking_sets(self):
        return combinations(self.elements(), self.blocking_threshold())
    
    def slices(self) -> set[set] :
        """
        Returns the set of all slices of this QSet.
        """
        def slices_of_level_1_slice(s):
            l = [e.slices() if isinstance(e, QSet) else frozenset([frozenset([e])]) for e in s]
            return {frozenset().union(*t) for t in product(*l)}
        return {s for s1 in self.level_1_slices() for s in slices_of_level_1_slice(s1)}
    
    def blocking_sets(self) -> set[set] :
        def v_blocking_of_level_1_blocking(b):
            l = [e.blocking_sets() if isinstance(e, QSet) else frozenset([frozenset([e])]) for e in b]
            return {frozenset().union(*t) for t in product(*l)}
        return {b for b1 in self.level_1_v_blocking_sets() for b in v_blocking_of_level_1_blocking(b1)}

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
    
    def to_dict(self):
        return {
            'threshold': self.threshold,
            'validators': list(self.validators),
            'innerQuorumSets': [iqs.to_dict() for iqs in self.inner_qsets]
        }

def qset_intersection_bound(qset1, qset2):
    """
    Returns a lower bound on the number of validators in common in any two slices of qset1 and qset2.
    Assumes that the qsets have at most 2 levels.
    Additionally, if qset1.validators and qset2.validators are empty and their inner qsets represent organizations,
    this is also a lower bound on the number of organizations one must compromise to obtain two slices that have only compromised
    organizations in common.
    """
    if qset1.depth() > 2 or qset2.depth() > 2:
        logging.warning("qsets have more than 2 levels; returning 0")
        return 0
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

@lru_cache(maxsize=None)
def qset_intersect(q1: QSet, q2: QSet) -> bool:
    """
    Brute-force check; may be slow.
    TODO: We could also follow the qset structures: check whether there is intersection at level 1, then, for each level-1 intersection, check that there is intersection at level 2, etc. That would probably be more efficient.
    """
    logging.info("Performing brute-force intersection-check of %s and %s", q1, q2)
    ss1 = q1.slices()
    ss2 = q2.slices()
    # now check whether every pair of slices intersects:
    return all(s1 & s2 for s1 in ss1 for s2 in ss2)


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

    # TODO: this is stellarbeat specific
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
            valid_validators = {
                v for v in new_qset_map if is_valid(v, new_qset_map)}
            if valid_validators == new_qset_map.keys():
                break
            new_qset_map = {v: qs for v,
                            qs in new_qset_map.items() if v in valid_validators}

        return FBAS(new_qset_map, self.metadata)

    # TODO also stellarbeat specific:
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
            if 'quorumSet' not in v or v['quorumSet'] is None:
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
    
    def to_json(self):
        return [dict(
            publicKey=k,
            quorumSet=v.to_json(),
            **self.metadata.get(k, {})
        ) for k, v in self.qset_map.items()]
    
    def restrict_to_reachable(self, v):
        """
        Returns a new FBAS with only the validators reachable from v.
        """
        reachable = nx.descendants(self.to_graph(), v) | {v}
        return FBAS({k: qs for k, qs in self.qset_map.items() if k in reachable}, self.metadata)

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

    def fast_intersection_check(self, v) -> Literal['true', 'unknown']:
        """
        This is a fast but heuristic method to check whether the set of validators reachable from v has quorum intersection.
        It may return 'unknown' even if the property holds; however it is sound: if it returns 'true', then the property holds.

        Typical examples where quorum-intersection holds but the heuristic fails to see it are circular FBASs like in cicular_1.json and circular_2.json
        """
        # first, compute what's reachable from v:
        g = self.to_graph()
        reachable = nx.descendants(g, v) | {v}
        logging.info("There are %s reachable validators", len(reachable))

        # next, compute the maximal sccs in the subgraph induced by the reachable validators
        # TODO: prioritize those that are closer to v?
        sccs = nx.strongly_connected_components(g.subgraph(reachable))
        # for each scc, check whether it contains an intertwined set whose closure covers all reachable validators.
        # check e.g. 10 sccs:
        for scc in islice(sccs, 10):
            # create an undirected graph over scc where there is an edge between v1 and v2 if and only if their QSets have a non-zero intersection bound:
            intertwined = nx.Graph()
            for v1, v2 in combinations(scc, 2):
                q1 = self.qset_map[v1]
                q2 = self.qset_map[v2]
                if v1 != v2 and (qset_intersection_bound(q1, q2) > 0 or qset_intersect(q1, q2)):
                    intertwined.add_edge(v1, v2)
            # compute the maximal cliques in the intertwined graph:
            cliques = nx.find_cliques(intertwined)
            # if the closure of one of those cliques is the whole set of reachable validators, we know that intersection holds.
            # check e.g. 10 cliques:
            for clique in islice(cliques, 10):
                # if the closure of the clique contains the reachable validators, we're done:
                if reachable <= self.closure(clique):
                    return 'true'
        return 'unknown'

    def splitting_set_bound_heuristic(self):
        pass

    def collapse_qsets(self) -> 'FBAS':
        raise NotImplementedError
   
    def all_have_meta_field(self, field : str) -> bool:
        return all(field in self.metadata[v] for v in self.validators())
    
    def meta_field_values(self, field : str) -> set[str]:
        return {self.metadata[v].get(field) for v in self.validators()} - {None}
    
    def validators_with_meta_field_value(self, field : str, value : str) -> set:
        return {v for v in self.validators() if self.metadata[v].get(field) == value}
    
    def validator_name(self, v) -> str:
        return (
            self.metadata.get(v).get('name')
                if self.metadata.get(v) and self.metadata.get(v).get('name')
            else str(v)
        )

    def format_qset(self, qset : QSet) -> str:
        def as_dict(qset):
            return {
                'threshold': qset.threshold,
                'validators': [self.validator_name(v) for v in qset.validators],
                'innerQuorumSets': [as_dict(iqs) for iqs in qset.inner_qsets]
            }
        return pformat(as_dict(qset), indent=2)