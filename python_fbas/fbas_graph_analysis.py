"""
SAT-based analysis of FBAS graphs
"""

import logging
import time
from typing import Any, Literal, Optional, Tuple, Collection
from itertools import combinations
from pysat.solvers import Solver
from pysat.examples.lsu import LSU # MaxSAT algorithm
from pysat.examples.rc2 import RC2 # MaxSAT algorithm
from pysat.formula import Or, And, Neg, Atom, Implies, Formula, WCNF
from pysat.card import CardEnc, EncType
from python_fbas.utils import to_cnf
from python_fbas.fbas_graph import FBASGraph

next_int = 1

def dnf_to_cnf(dnf: list[list[int]]) -> list[list[int]]:
    """
    Transforms a disjuntion of conjunctions (e.g. 2 out of 3 is (x1 and x2) or (x1 and x3) or (x2 and x3)) to CNF.
    We use the Tseitin method: we create one new variable for each conjunction and, for each new variable, we add clauses that enforce the equivalence between the new variable and the conjunction it represents.
    Finally we create a clauses that's the disjunction of all the new variables. This clause is the last in the returned list.
    """
    global next_int
    clauses:list[list[int]] = []
    for i, conj in enumerate(dnf):
        clauses.append([-v for v in conj] + [next_int+i])
        for v in conj:
            clauses.append([-(next_int+i), v])
    # finally, the top-level disjuntion:
    clauses.append([next_int + i for i in range(len(dnf))])
    # update next_int:
    next_int += len(dnf)
    return clauses

def card_constraint_to_cnf(ante: list[int], vs: Collection[int], threshold: int, card_encoding:Literal['naive','totalizer']='naive') -> list[list[int]]:
    """
    Given a set of variables vs, create a CNF formula that enforces that, if all vars in ante are true, then at least threshold variable in vs are true.
    """
    assert card_encoding in ['naive', 'totalizer']
    if card_encoding == 'naive':
        return card_constraint_to_cnf_naive(ante, vs, threshold)
    return card_constraint_to_cnf_totalizer(ante, vs, threshold)
    
def card_constraint_to_cnf_naive(ante: list[int], vs: Collection[int], threshold: int) -> list[list[int]]:
    """
    Given a set of variables vs, create a CNF formula that enforces that, if all vars in ante are true, then at least threshold variable in vs are true.
    As a propositional formula, this is a disjuntion of conjunctions (e.g. 2 out of 3 is (x1 and x2) or (x1 and x3) or (x2 and x3)).
    """
    terms = [list(conj) for conj in combinations(vs, threshold)]
    return [c+[-a for a in ante] for c in dnf_to_cnf(terms)]

def card_constraint_to_cnf_totalizer(ante: list[int], vs: Collection[int], threshold: int) -> list[list[int]]:
    """
    Given a set of variables vs, create a CNF formula that enforces that, if all vars in ante are true, then at least threshold variable in vs are true.
    Uses the totalizer encoding.
    """
    global next_int
    assert threshold > 0
    ante_neg = [-a for a in ante]
    if threshold == len(vs):
        return [ante_neg+[u] for u in vs]
    cnfp = CardEnc.atleast(lits=list(vs), bound=threshold, top_id=next_int, encoding=EncType.totalizer)
    next_int = cnfp.nv+1
    clauses = cnfp.clauses[:-1]
    clauses.append(cnfp.clauses[-1] + ante_neg)
    return clauses

def find_disjoint_quorums(fbas: FBASGraph, solver='cryptominisat5', flatten:bool=False, card_encoding:Literal['naive','totalizer']='naive') ->  Optional[Tuple[Collection, Collection]]:
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
        logging.error("Flattening is not supported with the new codebase")
        exit(1)

    start_time = time.time()

    global next_int
    next_int = 1
    # the clauses of the CNF formula:
    clauses: list[list[int]] = []

    # first, for each vertex in the FBAS graph, create two variables each indicating whether the vertex is in quorum A or quorum B.
    # also create a map to keep track of which variables encodes what.
    in_quorum_vars:dict[Any,int] = {}
    in_quorum_vars_inverse:dict[int,Any] = {}
    for q in ['A', 'B']:
        for v in fbas.vertices():
            in_quorum_vars[(q, v)] = next_int
            in_quorum_vars_inverse[next_int] = (q, v)
            next_int += 1
    def get_quorum(q:str, model:list[int]) -> list:
        return [in_quorum_vars_inverse[i][1] for i in model
                    if i in in_quorum_vars_inverse.keys() \
                        and in_quorum_vars_inverse[i][0] == q \
                        and in_quorum_vars_inverse[i][1] in fbas.validators]

    # now we create the constraints:
    for q in ['A', 'B']: # for each of our two quorums
        # first, we create a clause asserting that the quorum contains at least one validator:
        clauses.append([in_quorum_vars[(q, v)] for v in fbas.validators])
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [in_quorum_vars[(q, n)] for n in fbas.graph.successors(v)]
                clauses += card_constraint_to_cnf([in_quorum_vars[(q, v)]], vs, fbas.threshold(v), card_encoding=card_encoding)
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: # validators for which we don't have a threshold cannot be in the quorum:
                clauses.append([-in_quorum_vars[(q, v)]])
    # finally, we add the constraint that no validator can be in both quorums:
    for v in fbas.validators:
        clauses.append([-in_quorum_vars[('A', v)], -in_quorum_vars[('B', v)]])

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
        q1 = get_quorum('A', model)
        q2 = get_quorum('B', model)
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        return (q1, q2)
    
def find_minimal_splitting_set(fbas: FBASGraph, card_encoding:Literal['naive','totalizer']='naive') ->  Optional[Collection]:
    logging.info(f"Finding minimal-cardinality splitting set using MaxSAT with {card_encoding} cardinality encoding")

    # TODO: this could help, but it's tricky... in the maxsat problem, the logical validators should be assigned a weight corresponding to what they represent.
    # this will require tracking this in flatten_diamonds, and then using this information here.
    # fbas.flatten_diamonds()

    start_time = time.time()

    global next_int
    next_int = 1
    # the clauses of the CNF formula:
    clauses: list[list[int]] = []

    # first, for each vertex in the FBAS graph, create two variables each indicating whether the vertex is in quorum A or quorum B.
    # also create a map to keep track of which variables encodes what.
    in_quorum_vars = {}
    in_quorum_vars_inverse = {}
    for q in ['A', 'B']:
        for v in fbas.vertices():
            in_quorum_vars[(q, v)] = next_int
            in_quorum_vars_inverse[next_int] = (q, v)
            next_int += 1
    def get_quorum(q, positive_vars):
        return [in_quorum_vars_inverse[i][1] for i in positive_vars
                    if i in in_quorum_vars_inverse.keys() \
                        and in_quorum_vars_inverse[i][0] == q \
                        and in_quorum_vars_inverse[i][1] in fbas.validators]

    # create variables indicating whether a validator is faulty:
    is_faulty_vars = {}
    is_faulty_vars_inverse = {}
    for v in fbas.validators:
        is_faulty_vars[v] = next_int
        is_faulty_vars_inverse[next_int] = v
        next_int += 1

    # now we create the constraints:
    for q in ['A', 'B']: # for each of our two quorums
        # first, we create clauses asserting that the quorum contains at least one non-faulty validator:
        terms = [[in_quorum_vars[(q, v)], -is_faulty_vars[v]] for v in fbas.validators]
        clauses += dnf_to_cnf(terms)
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [in_quorum_vars[(q, n)] for n in fbas.graph.successors(v)]
                if v in fbas.validators:
                    # the threshold must be met only if the validator is not faulty:
                    clauses += card_constraint_to_cnf([in_quorum_vars[(q, v)], -is_faulty_vars[v]], vs, fbas.threshold(v), card_encoding=card_encoding)
                else:
                    # the threshold must be met:
                    clauses += card_constraint_to_cnf([in_quorum_vars[(q, v)]], vs, fbas.threshold(v), card_encoding=card_encoding)
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: # validators for which we don't have a threshold cannot be in the quorum:
                clauses.append([-in_quorum_vars[(q, v)]])
    # add the constraint that no non-faulty validator can be in both quorums:
    for v in fbas.validators:
        clauses.append([is_faulty_vars[v], -in_quorum_vars[('A', v)], -in_quorum_vars[('B', v)]])
    # finally, convert to weighted CNF and add soft constraints that minimize the number of faulty validators:
    wcnf = WCNF()
    wcnf.extend(clauses)
    for v in fbas.validators:
        wcnf.append([-is_faulty_vars[v]], weight=1)

    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    # now call the solver:
    s = LSU(wcnf)
    # s = RC2(wcnf)
    start_time = time.time()
    res = s.solve()
    # res = s.compute()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if not res:
        print("No splitting set found!")
        return None
    else:
        print(f"Found minimal-cardinality splitting set, size is {s.cost}")
        model = list(s.model)
        ss = [is_faulty_vars_inverse[i] for i in model if i in is_faulty_vars_inverse.keys()]
        logging.info("Minimal-cardinality splitting set: %s", ss)
        q1 = get_quorum('A', model)
        q2 = get_quorum('B', model)
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        return ss
    
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