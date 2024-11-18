"""
SAT-based analysis of FBAS graphs with Z3
"""

import logging
import time
from typing import Collection
from itertools import combinations
from z3 import And, Or, Solver, Bool, Not, sat, Implies, BoolRef, BoolVal

from python_fbas.fbas_graph import FBASGraph

def find_disjoint_quorums(fbas: FBASGraph) -> bool:
    """
    Try to find two disjoint quorums in the FBAS graph, or prove that they don't exist.
    TODO: return the disjoint quorums if found
    """
    logging.info("Finding disjoint quorums with Z3")

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
    
    def set_in_quorum(s: Collection, q : str) -> list:
        return And(*[in_quorum(q, n) for n in s])
    
    def quorum_satisfies_requirements(n: str, q: str) -> BoolRef:
        if fbas.threshold(n) > 0:
            return Or(*[set_in_quorum(s, q)
                for s in combinations(fbas.graph.successors(n), fbas.threshold(n))])
        elif fbas.threshold(n) == 0:
            return BoolVal(True)
        else:
            return BoolVal(False)
    
    start_time = time.time()
    constraints : list[BoolRef] = []
    for q in ['A', 'B']: # our two quorums
        # the quorum must be non-empty:
        constraints += [Or(*[in_quorum(q, n) for n in fbas.validators])]
        # the quorum must satisfy the requirements of each of its members:
        constraints += \
            [Implies(in_quorum(q, n), quorum_satisfies_requirements(n, q))
                for n in fbas.graph.nodes()]
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
        logging.info("Found disjoint quorums")
        q1 = [n for n in fbas.validators if s.model()[in_quorum('A', n)]]
        q2 = [n for n in fbas.validators if s.model()[in_quorum('B', n)]]
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        return True
    return False
