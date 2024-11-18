"""
SAT-based analysis of FBAS graphs
"""

import logging
import time
from typing import Optional, Tuple, Collection
from itertools import combinations, product
from pysat.solvers import Solver
from pysat.formula import Or, And, Neg, Atom, Implies, Formula
from python_fbas.utils import to_cnf

from python_fbas.fbas_graph import FBASGraph

def find_disjoint_quorums(fbas: FBASGraph, solver='cms', flatten=False) -> Optional[Tuple[Collection, Collection]]:
    """
    Try to find two disjoint quorums in the FBAS graph, or prove that they don't exist.

    We create a formula that is satisfiable if and only if there exist two disjoint quorums in the (flattened) FBAS.
    The idea is to create two propositional variables ('A',v) and ('B',v) for each validator v.
    ('A',v) indicates whether v is in a quorum 'A', and ('B',v) indicates whether v is in a quorum 'B'.
    Then we create constraints asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for satisfiability.
    If the constraints are satisfiable, then we have two disjoint quorums.
    Otherwise, we know that no two disjoint quorums exist.

    TODO: This is slow due to the way pysat builds formulas (most of the time is spent building string representations of formulas for hashing...).
    We could try our own CNF encoding.
    
    TODO: seems like a manual encoding following the totalizer encoding would be good.
    """
    logging.info("Finding disjoint quorums with pysat")

    def in_quorum(q, n):
        return Atom((q, n))

    def get_quorum_from_atoms(atoms, q) -> list:
        """Given a list of SAT atoms, return the validators in quorum q."""
        return [a.object[1] for a in atoms
                if isinstance(a, Atom) and a.object[0] == q and a.object[1] in fbas.validators]
    
    def set_in_quorum(s: Collection, q : str) -> Formula:
        return And(*[in_quorum(q, n) for n in s])
    
    def quorum_satisfies_requirements(n: str, q: str) -> Formula:
        if fbas.threshold(n) > 0:
            return Or(*[set_in_quorum(s, q)
                for s in combinations(fbas.graph.successors(n), fbas.threshold(n))])
        elif fbas.threshold(n) == 0:
            return And()
        else:
            return Or()

    if flatten:
        start_time = time.time()
        fbas.flatten_diamonds()
        end_time = time.time()
        logging.info("flattening time: %s", end_time - start_time)

    start_time = time.time()
    constraints : list[Formula] = []
    for q in ['A', 'B']: # our two quorums
        # the quorum must be non-empty:
        constraints += [Or(*[in_quorum(q, n) for n in fbas.validators])]
        # the quorum must satisfy the requirements of each of its members:
        constraints += \
            [Implies(in_quorum(q, n), quorum_satisfies_requirements(n, q))
                for n in fbas.graph.nodes()]
    # no validator can be in both quorums:
    for v in fbas.validators:
        constraints += [Neg(And(in_quorum('A', v), in_quorum('B', v)))]
    clauses = to_cnf(constraints)
    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    # now call the solver:
    s = Solver(bootstrap_with=clauses, name=solver)
    start_time = time.time()
    res = s.solve()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if res:
        model = s.get_model()
        fmlas = list(Formula.formulas(model, atoms_only=True))
        q1 = get_quorum_from_atoms(fmlas, 'A')
        q2 = get_quorum_from_atoms(fmlas, 'B')
        logging.info("Disjoint quorums found")
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        assert fbas.is_quorum(q1)
        assert fbas.is_quorum(q2)
        assert not set(q1) & set(q2)
        return (q1, q2)
    return None

def find_disjoint_quorums_cnf(fbas: FBASGraph, solver='cms') ->  Optional[Tuple[Collection, Collection]]:
    """
    Try to find two disjoint quorums in the FBAS graph, or prove that they don't exist. Directly generate the CNF formula.
    """

    start_time = time.time()
    clauses: list[list[int]] = []

    # first, we create two variables per vertex:
    # ('A',v) indicates whether v is in quorum A
    # ('B',v) indicates whether v is in quorum B
    pair_to_int = {}
    int_to_pair = {}
    next_int = 1
    logging.debug("number of vertices: %s", len(fbas.vertices()))
    for q in ['A', 'B']:
        for v in fbas.vertices():
            pair_to_int[(q, v)] = next_int
            int_to_pair[next_int] = (q, v)
            next_int += 1
    logging.debug("finished creating variables")

    for q in ['A', 'B']:
        # first, we create a clause asserting that the quorum contains at least one validator:
        clauses.append([pair_to_int[(q, v)] for v in fbas.validators])
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                # TODO: this seems correct but is way too slow (exponential!); better do Tseitin...
                logging.debug("Threshold of %s is %s out of %s", v, fbas.threshold(v), fbas.graph.out_degree(v))
                children_vars = [pair_to_int[(q, u)] for u in fbas.graph.successors(v)]
                card_t_sets = combinations(children_vars, fbas.threshold(v))
                card_t_clauses = product(*card_t_sets)
                for clause in card_t_clauses:
                    clauses.append(list(set(clause)) + [-pair_to_int[(q,v)]])
            if fbas.threshold(v) < 0: # validators for which we don't have a threshold cannot be in the quorum:
                clauses.append([-pair_to_int[(q, v)]])
    # finally, we add the constraint that no validator can be in both quorums:
    for v in fbas.validators:
        clauses.append([-pair_to_int[('A', v)], -pair_to_int[('B', v)]])
    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    # now call the solver:
    s = Solver(name=solver)
    s = Solver(bootstrap_with=clauses, name=solver)
    start_time = time.time()
    res = s.solve()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if not res:
        return None
    else:
        print("Found disjoint quorums!")
        model = s.get_model()
        def get_quorum(q):
            return [int_to_pair[i][1] for i in model
                        if i in int_to_pair.keys() \
                            and int_to_pair[i][0] == q \
                            and int_to_pair[i][1] in fbas.validators]
        q1 = get_quorum('A')
        q2 = get_quorum('B')
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        return (q1, q2)