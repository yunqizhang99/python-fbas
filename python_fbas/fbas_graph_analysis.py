"""
SAT-based analysis of FBAS graphs
"""

import logging
import time
from typing import Optional, Tuple
from itertools import combinations
from pysat.solvers import Solver
from pysat.formula import Or, And, Neg, Atom, Implies, Formula
from python_fbas.utils import to_cnf

from python_fbas.fbas_graph import FBASGraph

def find_disjoint_quorums(fbas: FBASGraph, solver='cms', flatten=False) -> Optional[Tuple[set, set]]:
    """
    Try to find two disjoint quorums in the FBAS graph, or prove that they don't exist.

    First we "flatten all the diamonds" in the FBAS (makes things simpler).
    Then we create a formula that is satisfiable if and only if there exist two disjoint quorums in the (flattened) FBAS.
    The idea is to create two propositional variables ('A',v) and ('B',v) for each validator v.
    ('A',v) indicates whether v is in a quorum 'A', and ('B',v) indicates whether v is in a quorum 'B'.
    Then we create constraints asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for satisfiability.
    If the constraints are satisfiable, then we have two disjoint quorums.
    Otherwise, we know that no two disjoint quorums exist.
    """

    def in_quorum(q, n):
        return Atom((q, n))
    
    def get_quorum_from_atoms(atoms, q):
        """Given a list of SAT atoms, return the validators in quorum q."""
        return [a.object[1] for a in atoms
                if isinstance(a, Atom) and a.object[0] == q and a.object[1] in fbas.validators]

    if flatten:
        start_time = time.time()
        fbas.flatten_diamonds()
        end_time = time.time()
        logging.info("flattening time: %s", end_time - start_time)

    start_time = time.time()
    constraints : list[Formula] = []
    for q in ['A', 'B']: # our two quorums
        # q is not empty:
        constraints += [Or(*[in_quorum(q,v) for v in fbas.validators])]
        # the quorum must satisfy the requirements of each of its members:
        constraints += \
            [Implies(in_quorum(q, n), And(*[in_quorum(q, c) for c in slice]))
                for n in fbas.graph.nodes() if fbas.threshold(n) > 0
                for slice in combinations(fbas.graph.successors(n), fbas.threshold(n))]
        # do not include validators that do not have a qset:
        constraints += [Neg(in_quorum(q, v)) for v in fbas.validators if fbas.threshold(v) == -1]
    # no validator can be in both quorums:
    for v in fbas.validators:
        constraints += [Neg(And(in_quorum('A', v), in_quorum('B', v)))]
    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    # now solve the constraints:
    clauses = to_cnf(constraints)
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
        return (q1, q2)
    return None
