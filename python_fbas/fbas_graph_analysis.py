"""
SAT-based analysis of FBAS graphs
"""

import logging
import time
from typing import Optional, Tuple, Collection
from itertools import combinations
from pysat.solvers import Solver
from pysat.formula import Or, And, Neg, Atom, Implies, Formula
from python_fbas.utils import to_cnf
from python_fbas.fbas_graph import FBASGraph

def find_disjoint_quorums(fbas: FBASGraph, solver='cryptominisat5', flatten=False) ->  Optional[Tuple[Collection, Collection]]:
    """
    Find two disjoint quorums in the FBAS graph, or prove there are none.
    To do this, we build a CNF formula that is satsifiable if and only if there are two disjoint quorums.
    Then we call a SAT solver to check for satisfiability.

    The idea is to consider two quorums A and B and create two propositional variables vA and vB for each validator v, where vA is true when v is in quorum A, and vB is true when v is in quorum B.
    Then we create constraints asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for satisfiability.
    If the constraints are satisfiable, then we have two disjoint quorums and the truth assignment gives us the quorums.
    Otherwise, we know that no two disjoint quorums exist.
    """
    logging.info("Finding disjoint quorums with solver %s", solver)

    if flatten:
        fbas.flatten_diamonds() # this is not very useful

    start_time = time.time()

    # integer counter used to create new variables:
    next_int = 1
    # the clauses of the CNF formula:
    clauses: list[list[int]] = []

    def card_constraint_to_cnf(vs: Collection[int], threshold: int) -> list[list[int]]:
        """
        Given a set of variables vs, create a CNF formula that enforces that at least `threshold` of them are true.
        As a propositional formula, this is a disjuntion of conjunctions (e.g. 2 out of 3 is (x1 and x2) or (x1 and x3) or (x2 and x3)).

        We transform this to CNF using the Tseitin method: we create one new variable for each conjunction and, for each new variable, we add clauses that enforce the equivalence between the new variable and the conjunction it represents.
        Finally we create a clauses that's the disjunction of all the new variables. This clause is the last in the returned list.

        Note that variables are just integers, and their negation is the negative of the integer.

        TODO: try the totalizer encoding, which should be even more efficient.
        """
        nonlocal next_int
        clauses:list[list[int]] = []
        card_t_sets = list(combinations(vs, threshold))
        # we create len(card_t_sets) auxiliary variables, one for each set:
        for i, card_t_set in enumerate(card_t_sets):
            # the conjunction of the elements in the set implies the set's variable:
            clauses.append([-v for v in card_t_set] + [next_int+i])
            # for each element in the set, the set's variable implies the element:
            for v in card_t_set:
                clauses.append([-(next_int+i), v])
        # finally, add the top-level disjuntion:
        clauses.append([next_int + i for i in range(len(card_t_sets))])
        # update next_int:
        next_int += len(card_t_sets)
        return clauses

    # first, for each vertex in the FBAS graph, create two variables each indicating whether the vertex is in quorum A or quorum B.
    # also create a map to keep track of which variables encodes what.
    pair_to_int = {}
    int_to_pair = {}
    for q in ['A', 'B']:
        for v in fbas.vertices():
            pair_to_int[(q, v)] = next_int
            int_to_pair[next_int] = (q, v)
            next_int += 1

    # now we create the constraints:
    for q in ['A', 'B']: # for each of our two quorums
        # first, we create a clause asserting that the quorum contains at least one validator:
        clauses.append([pair_to_int[(q, v)] for v in fbas.validators])
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [pair_to_int[(q, n)] for n in fbas.graph.successors(v)]
                card_clauses = card_constraint_to_cnf(vs, fbas.threshold(v))
                # add all but the last clause:
                clauses += card_clauses[:-1]
                # the current variable implies the cardinality constraint:
                clauses.append(card_clauses[-1] + [-pair_to_int[(q, v)]])
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: # validators for which we don't have a threshold cannot be in the quorum:
                clauses.append([-pair_to_int[(q, v)]])
    # finally, we add the constraint that no validator can be in both quorums:
    for v in fbas.validators:
        clauses.append([-pair_to_int[('A', v)], -pair_to_int[('B', v)]])

    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    # now call the solver:
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
    
def find_disjoint_quorums_using_pysat_fmla(fbas: FBASGraph, solver='cms', flatten=False) -> Optional[Tuple[Collection, Collection]]:
    """
    Similar to find_disjoint_quorums, but encodes the problem to pysat's Formula class. Unfortunately this is very slow (most of the time is spent building pysat formulas).
    """
    logging.info("Finding disjoint quorums by encoding to a pysat Formula")

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
    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)
    start_time = time.time()
    clauses = to_cnf(constraints)
    end_time = time.time()
    logging.info("Time to convert to CNF: %s", end_time - start_time)

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