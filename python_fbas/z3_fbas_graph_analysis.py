"""
SAT-based analysis of FBAS graphs with Z3
"""

import logging
import time
from typing import Optional, Tuple, Collection
from itertools import combinations
from z3 import And, Or, Solver, Bool, Not, sat, Implies, BoolRef

from python_fbas.fbas_graph import FBASGraph

def find_disjoint_quorums(fbas: FBASGraph) -> Optional[Tuple[set, set]]:
    """
    Try to find two disjoint quorums in the FBAS graph, or prove that they don't exist.
    """

    var_count = 0
    variables = {}

    def in_quorum(q, n) -> BoolRef:
        """
        Return a variable indicating whether n is in quorum q
        """
        assert isinstance(q, str) and isinstance(n, str)
        nonlocal var_count
        nonlocal variables
        if (q, n) not in variables:
            variables[(q, n)] = Bool(var_count)
            var_count += 1
        return variables[(q, n)]
    
    def slice_satisfied_by_quorum(s: Collection, q : str) -> list:
        return [And(*[in_quorum(q, n) for n in s])]

    def get_quorum_from_atoms(atoms, q) -> list:
        """Given a list of SAT atoms, return the validators in quorum q."""
        pass

    start_time = time.time()
    constraints : list[BoolRef] = []
    for q in ['A', 'B']: # our two quorums
        # q has at least one validator for which we have a qset:
        constraints += [Or(*[in_quorum(q,v) for v in fbas.validators if fbas.threshold(v) > -1])]
        # the quorum must satisfy the requirements of each of its members:
        constraints += \
            [Implies(in_quorum(q, n), Or(*slice_satisfied_by_quorum(s, q)))
                for n in fbas.graph.nodes() if fbas.threshold(n) > 0
                for s in combinations(fbas.graph.successors(n), fbas.threshold(n))]
    # no validator can be in both quorums:
    for v in fbas.validators:
        constraints += [Not(And(in_quorum('A', v), in_quorum('B', v)))]
    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    # now call the solver:
    s = Solver()
    s.add(*constraints)
    start_time = time.time()
    res = s.check()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if res == sat:
        model = s.model()
        logging.info("Found disjoint quorums")
        return (set(), set())
    return None
