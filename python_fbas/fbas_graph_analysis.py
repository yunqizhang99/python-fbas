"""
SAT-based analysis of FBAS graphs

TODO: Most of the runtime is spent building pysat formulas. Try the Z3 python bindings to see if it's faster.
"""

import logging
import time
from typing import Optional, Tuple, Collection
from itertools import combinations
from pysat.solvers import Solver
from pysat.formula import Or, And, Neg, Atom, Implies, Formula, PYSAT_FALSE, PYSAT_TRUE
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
