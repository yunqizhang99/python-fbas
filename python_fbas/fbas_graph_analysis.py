"""
SAT-based analysis of FBAS graphs
"""

import logging
import time
from typing import Any, Optional, Tuple, Collection
from itertools import combinations
import networkx as nx
from pysat.solvers import Solver
from pysat.formula import CNF, WCNF
from pysat.examples.lsu import LSU # MaxSAT algorithm
from pysat.examples.rc2 import RC2 # MaxSAT algorithm
from python_fbas.fbas_graph import FBASGraph
import python_fbas.propositional_logic as pl
import python_fbas.config as config

# TODO: implement grouping (e.g. by homeDomain)

def find_disjoint_quorums(fbas: FBASGraph) -> Optional[Tuple[Collection, Collection]]:
    """
    Find two disjoint quorums in the FBAS graph, or prove there are none.
    To do this, we build a propositional formula that is satsifiable if and only if there are two disjoint quorums.
    Then we call a SAT solver to check for satisfiability.

    The idea is to consider two quorums A and B and create two propositional variables vA and vB for each validator v, where vA is true when v is in quorum A, and vB is true when v is in quorum B.
    Then we create constraints asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for satisfiability.
    If the constraints are satisfiable, then we have two disjoint quorums and the truth assignment gives us the quorums.
    Otherwise, we know that no two disjoint quorums exist.
    """

    quorum_tag:int = 1

    def in_quorum(q:str, n:str) -> pl.Atom:
        """Returns an atom denoting whether node n is in quorum q."""
        return pl.Atom((quorum_tag, q, n))

    def get_quorum_(atoms:list[int], q:str, fbas:FBASGraph) -> list[str]:
        """Given a list of atoms, returns the validators in quorum q."""
        return [pl.variables_inv[v][2] for v in set(atoms) & set(pl.variables_inv.keys())
                if pl.variables_inv[v][0] == quorum_tag and pl.variables_inv[v][1] == q
                    and pl.variables_inv[v][2] in fbas.validators]
    
    start_time = time.time()
    constraints : list[pl.Formula] = []
    for q in ['A', 'B']: # our two quorums
        # the quorum must be non-empty:
        constraints += [pl.Or(*[in_quorum(q, n) for n in fbas.validators])]
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [in_quorum(q, n) for n in fbas.graph.successors(v)]
                constraints.append(pl.Implies(in_quorum(q, v), pl.Card(fbas.threshold(v), *vs)))
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: # validators for which we don't have a threshold cannot be in the quorum:
                constraints.append(pl.Not(in_quorum(q, v)))
    # no validator can be in both quorums:
    for v in fbas.validators:
        constraints += [pl.Or(pl.Not(in_quorum('A', v)), pl.Not(in_quorum('B', v)))]
    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)
    start_time = time.time()
    clauses = pl.to_cnf(constraints)
    end_time = time.time()
    logging.info("Time to convert to CNF: %s", end_time - start_time)

    # now call the solver:
    s = Solver(bootstrap_with=clauses, name=config.sat_solver)
    start_time = time.time()
    res = s.solve()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)

    if config.output:
        if config.output:
            with open(config.output, 'w', encoding='utf-8') as f:
                cnf = CNF(from_clauses=clauses)
                dimacs = cnf.to_dimacs()
                comment = "c " + ("SATISFIABLE" if res else "UNSATISFIABLE") + "\n"
                f.write(comment)
                f.writelines(dimacs)

    if res:
        model = s.get_model()
        q1 = get_quorum_(model, 'A', fbas)
        q2 = get_quorum_(model, 'B', fbas)
        logging.info("Disjoint quorums found")
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        assert fbas.is_quorum(q1)
        assert fbas.is_quorum(q2)
        assert not set(q1) & set(q2)
        return (q1, q2)
    return None
    
def maximize(wcnf:WCNF) -> Optional[Tuple[int, Any]]:
    """
    Solve a MaxSAT CNF problem.
    """
    if config.max_sat_algo == 'LSU':
        s = LSU(wcnf)
    else:
        s = RC2(wcnf)
    start_time = time.time()
    if config.max_sat_algo == 'LSU':
        res = s.solve()
    else:
        res = s.compute()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if res:
        return s.cost, s.model
    return None

def find_minimal_splitting_set(fbas: FBASGraph) ->  Optional[Tuple[Collection,Collection,Collection]]:
    """
    Find a minimal-cardinality splitting set in the FBAS graph, or prove there is none.
    Uses one of pysat's MaxSAT procedures (LRU or RC2).
    If found, returns the splitting set and the two quorums that it splits.
    """

    logging.info("Finding minimal-cardinality splitting set using MaxSAT algorithm %s with %s cardinality encoding", config.max_sat_algo, config.card_encoding)

    faulty_tag:int = 0
    quorum_tag:int = 1

    def in_quorum(q:str, n:str) -> pl.Atom:
        """Returns an atom denoting whether node n is in quorum q."""
        return pl.Atom((quorum_tag, q, n))

    def get_quorum_(atoms:list[int], q:str, fbas:FBASGraph) -> list[str]:
        """Given a list of atoms, returns the validators in quorum q."""
        return [pl.variables_inv[v][2] for v in set(atoms) & set(pl.variables_inv.keys())
                if pl.variables_inv[v][0] == quorum_tag and pl.variables_inv[v][1] == q
                    and pl.variables_inv[v][2] in fbas.validators]

    def faulty(n:str) -> pl.Atom:
        """Returns an atom denoting whether node n is faulty."""
        return pl.Atom((faulty_tag,n))

    def get_faulty(atoms:list[int]) -> list[str]:
        """Given a list of atoms, returns the faulty validators."""
        return [pl.variables_inv[v][1] for v in set(atoms) & set(pl.variables_inv.keys()) \
                if pl.variables_inv[v][0] == faulty_tag]
    
    start_time = time.time()
    constraints : list[pl.Formula] = []

    # now we create the constraints:
    for q in ['A', 'B']: # for each of our two quorums
        # the quorum contains at least one non-faulty validator:
        constraints += [pl.Or(*[pl.And(in_quorum(q, n), pl.Not(faulty(n))) for n in fbas.validators])]
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [in_quorum(q, n) for n in fbas.graph.successors(v)]
                if v in fbas.validators:
                    # the threshold must be met only if the validator is not faulty:
                    constraints.append(pl.Implies(pl.And(in_quorum(q, v), pl.Not(faulty(v))), pl.Card(fbas.threshold(v), *vs)))
                else:
                    # the threshold must be met:
                    constraints.append(pl.Implies(in_quorum(q, v), pl.Card(fbas.threshold(v), *vs)))
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: # validators for which we don't have a threshold cannot be in the quorum:
                constraints.append(pl.Not(in_quorum(q, v)))
    # add the constraint that no non-faulty validator can be in both quorums:
    for v in fbas.validators:
        constraints += [pl.Or(faulty(v), pl.Not(in_quorum('A', v)), pl.Not(in_quorum('B', v)))]
    # finally, convert to weighted CNF and add soft constraints that minimize the number of faulty validators:
    wcnf = WCNF()
    wcnf.extend(pl.to_cnf(constraints))
    for v in fbas.validators:
        wcnf.append(pl.to_cnf(pl.Not(faulty(v)))[0], weight=1)

    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    result = maximize(wcnf)

    if not result:
        print("No splitting set found!")
        return None
    else:
        cost, model = result
        print(f"Found minimal-cardinality splitting set, size is {cost}")
        model = list(model)
        ss = get_faulty(model)
        logging.info("Minimal-cardinality splitting set: %s", [fbas.with_name(s) for s in ss])
        q1 = get_quorum_(model, 'A', fbas)
        q2 = get_quorum_(model, 'B', fbas)
        logging.info("Quorum A: %s", [fbas.with_name(v) for v in q1])
        logging.info("Quorum B: %s", [fbas.with_name(v) for v in q2])
        return (ss, q1, q2)

def find_minimal_blocking_set(fbas: FBASGraph) -> Optional[Collection[str]]:
    """
    Find a minimal-cardinality blocking set in the FBAS graph, or prove there is none.
    """

    logging.info("Finding minimal-cardinality blocking set using MaxSAT algorithm %s with %s cardinality encoding", config.max_sat_algo, config.card_encoding)

    start_time = time.time()

    constraints : list[pl.Formula] = []

    faulty_tag:int = 0
    blocked_tag:int = 1

    def is_faulty(v:str) -> pl.Atom:
        return pl.Atom((faulty_tag,v))
    
    def blocked(v:str) -> pl.Atom: # we should not need this (since all need to be blocked)
        return pl.Atom((blocked_tag,v))

    def lt(v1:str, v2:str) -> pl.Formula:
        """
        v1 is strictly lower than v2
        """
        return pl.Atom((v1, v2))
    
    def blocking_threshold(v) -> int:
        return len(list(fbas.graph.successors(v))) - fbas.threshold(v) + 1
    
    # first, the threshold constraints:
    for v in fbas.vertices():
        constraints.append(pl.Or(is_faulty(v), blocked(v)))
        if v not in fbas.validators:
            constraints.append(pl.Not(is_faulty(v)))
        if fbas.threshold(v) > 0:
            may_block = [pl.And(blocked(n), lt(n,v)) for n in fbas.graph.successors(v)]
            constraints.append(pl.Implies(pl.Card(blocking_threshold(v), *may_block), blocked(v)))
            constraints.append(pl.Implies(pl.And(blocked(v), pl.Not(is_faulty(v))), pl.Card(blocking_threshold(v), *may_block)))
        if fbas.threshold(v) < 0:
            constraints.append(pl.Not(blocked(v)))

    # the lt relation must be a partial order (anti-symmetric and transitive) on the vertices that matter:
    sccs = [scc for scc in nx.strongly_connected_components(fbas.graph)
            if fbas.is_quorum([v for v in set(scc) & fbas.validators])] # NOTE contains the union of all minimal quorums
    assert sccs
    k = set().union(*sccs)
    for v1 in k:
        constraints.append(pl.Not(lt(v1, v1)))
        for v2 in k:
            for v3 in k:
                constraints.append(pl.Implies(pl.And(lt(v1, v2), lt(v2, v3)), lt(v1, v3)))
    
    # convert to weighted CNF and add soft constraints that minimize the number of faulty validators:
    wcnf = WCNF()
    wcnf.extend(pl.to_cnf(constraints))
    for v in fbas.validators:
        wcnf.append(pl.to_cnf(pl.Not(is_faulty(v)))[0], weight=1)

    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    result = maximize(wcnf)

    if not result:
        print("No blocking set found!")
        return None
    else:
        cost, model = result
        model = list(model)
        logging.info("Found minimal-cardinality blocking set, size is %s", cost)
        s:list[str] = [pl.variables_inv[v][1] for v in set(model) & set(pl.variables_inv.keys()) \
            if pl.variables_inv[v][0] == faulty_tag]
        logging.info("Minimal-cardinality blocking set: %s", [fbas.with_name(v) for v in s])
        # model_debug = [(v, pl.variables_inv[abs(v)]) for v in set(model) & set(pl.variables_inv.keys())]
        # logging.info("model debug: %s", model_debug)
        assert fbas.closure(s) == fbas.validators
        for vs in combinations(s, cost-1):
            assert fbas.closure(vs) != fbas.validators
        return s
