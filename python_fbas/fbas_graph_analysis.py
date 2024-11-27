"""
SAT-based analysis of FBAS graphs
"""

import logging
import time
from typing import Any, Optional, Tuple, Collection
from itertools import combinations
from pysat.solvers import Solver
from pysat.examples.lsu import LSU # MaxSAT algorithm
from pysat.examples.rc2 import RC2 # MaxSAT algorithm
from pysat.formula import Or, And, Neg, Atom, Implies, Formula, WCNF, CNF
from pysat.card import CardEnc, EncType
from python_fbas.utils import to_cnf
from python_fbas.fbas_graph import FBASGraph
import python_fbas.propositional_logic as pl
import python_fbas.config as config

# A CNF formulat is a list of clauses, where a clause is a list of literals, where a literal is an integer denoting a propositional variable p > 0 or its negation -p.
Clauses = list[list[int]]

# TODO: move stuff to cnf_utils.py
# Use a context object to keep track of the global counter used to create fresh propositional variables; or pass a top_id around

next_int: int = 1 # global counter used to create fresh propositional variables

# TODO: Create low-level functions to encode constraints to CNF, i.e. classes for And, Or, Implies, and CNF conversion. Keep it simple and fast.

def dnf_to_cnf(dnf: Clauses) -> Clauses:
    """
    Transforms a disjuntion of conjunctions (e.g. 2 out of 3 is (x1 and x2) or (x1 and x3) or (x2 and x3)) to CNF.
    We use the Tseitin method: we create one new variable for each conjunction and, for each new variable, we add clauses that enforce the equivalence between the new variable and the conjunction it represents.
    Finally we create a clauses that's the disjunction of all the new variables. This clause is the last in the returned list.
    """
    global next_int
    clauses:Clauses = []
    for i, conj in enumerate(dnf):
        assert conj # conjunctions must be non-empty
        clauses.append([-v for v in conj] + [next_int+i])
        for v in conj:
            clauses.append([-(next_int+i), v])
    # finally, the top-level disjuntion:
    clauses.append([next_int + i for i in range(len(dnf))])
    # update next_int:
    next_int += len(dnf)
    return clauses

def negate_cnf(cnf: Clauses) -> Clauses:
    """
    Negate a CNF formula.
    """
    neg_dnf = [[-l for l in c] for c in cnf]
    return dnf_to_cnf(neg_dnf)

def card_constraint_to_cnf(ante: list[int], vs: Collection[int], threshold: int) -> Clauses:
    """
    Given a set of variables vs, create a CNF formula that enforces that, if all vars in ante are true, then at least threshold variable in vs are true.
    """
    if config.card_encoding == 'naive':
        return card_constraint_to_cnf_naive(ante, vs, threshold)
    elif config.card_encoding == 'totalizer':
        return card_constraint_to_cnf_totalizer(ante, vs, threshold)
    logging.error("Unknown cardinality encoding: %s", config.card_encoding)
    exit(1)
    
def card_constraint_to_cnf_naive(ante: list[int], vs: Collection[int], threshold: int) -> Clauses:
    """
    Given a set of variables vs, create a CNF formula that enforces that, if all vars in ante are true, then at least threshold variable in vs are true. We just naively enumerate all possibilities.
    As a propositional formula, this is a disjuntion of conjunctions (e.g. 2 out of 3 is (x1 and x2) or (x1 and x3) or (x2 and x3)).
    """
    terms = [list(conj) for conj in combinations(vs, threshold)]
    return [c+[-a for a in ante] for c in dnf_to_cnf(terms)]

def card_constraint_to_cnf_totalizer(ante: list[int], vs: Collection[int], threshold: int) -> Clauses:
    """
    Given a set of variables vs, create a CNF formula that enforces that, if all vars in ante are true, then at least threshold variable in vs are true.
    Uses the totalizer encoding.

    NOTE: relies on the fact that, in the list of clauses returned by pysat's totalizer encoding, the last clause is the one enforcing the threshold.
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

# TODO: version of card constraint for blocking set (this time the constraints imply something)

def make_quorum_vars(q:str, fbas:FBASGraph) -> Tuple[dict[Any,int], dict[int,Any]]:
    """
    Create variables denoting membership in the quorum q
    """
    global next_int
    quorum_vars:dict[Any,int] = {}
    quorum_vars_inverse:dict[int,Any] = {}
    for v in fbas.vertices():
        quorum_vars[(q, v)] = next_int
        quorum_vars_inverse[next_int] = (q, v)
        next_int += 1
    return (quorum_vars, quorum_vars_inverse)

def get_quorum(fbas:FBASGraph, q:str, model:list[int], quorum_vars_inverse:dict[int, str]) -> list[str]:
    """
    Extract quorum from a model (i.e. a list of literals)
    """
    return [quorum_vars_inverse[i][1] for i in model
                if i in quorum_vars_inverse.keys() \
                    and quorum_vars_inverse[i][0] == q \
                    and quorum_vars_inverse[i][1] in fbas.validators]

def find_disjoint_quorums(fbas: FBASGraph) ->  Optional[Tuple[Collection, Collection]]:
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

    # TODO: start with the heuristic (maybe upon user request)
    if config.heuristic_first:
        logging.info("Checking quorum intersection using the fast heuristic")
        res = fbas.fast_intersection_check()
        if res == 'true':
            print("All quorums intersect")
            exit(0)
        else:
            logging.info("Fast heuristic returned 'unknown'")

    logging.info("Finding disjoint quorums with solver %s", config.sat_solver)

    start_time = time.time()

    # the clauses of the CNF formula:
    clauses: Clauses = []

    # first, for each vertex in the FBAS graph, create two variables each indicating whether the vertex is in quorum A or quorum B.
    # also create a map to keep track of which variables encodes what.
    # TODO: method create_quorum_vars
    quorum_vars:dict[Any,int] = {}
    quorum_vars_inverse:dict[int,Any] = {}
    for q in ['A', 'B']:
        vars_, inverse = make_quorum_vars(q, fbas)
        quorum_vars.update(vars_)
        quorum_vars_inverse.update(inverse)

    # now we create the constraints:
    for q in ['A', 'B']: # for each of our two quorums
        # first, we create a clause asserting that the quorum contains at least one validator:
        clauses.append([quorum_vars[(q, v)] for v in fbas.validators])
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [quorum_vars[(q, n)] for n in fbas.graph.successors(v)]
                clauses += card_constraint_to_cnf([quorum_vars[(q, v)]], vs, fbas.threshold(v))
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: # validators for which we don't have a threshold cannot be in the quorum:
                clauses.append([-quorum_vars[(q, v)]])
    # finally, we add the constraint that no validator can be in both quorums:
    for v in fbas.validators:
        clauses.append([-quorum_vars[('A', v)], -quorum_vars[('B', v)]])

    end_time = time.time()
    logging.info("Constraint-building time: %s", end_time - start_time)

    if config.output:
        logging.info("Writing CNF formula to file %s", config.output)
        cnf = CNF(from_clauses=clauses)
        cnf.to_file(config.output)

    # now call the solver:
    s = Solver(bootstrap_with=clauses, name=config.sat_solver)
    start_time = time.time()
    res = s.solve()
    end_time = time.time()
    solving_time = end_time - start_time
    logging.info("Solving time: %s", solving_time)

    if config.output:
        # add comment indicating whether the formula is satisfiable or not
        # prepend comment to file:
        if config.output:
            with open(config.output, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            with open(config.output, 'w', encoding='utf-8') as f:
                comment = "c " + ("SATISFIABLE" if res else "UNSATISFIABLE") + " (cryptominisat5 runtime: " + str(solving_time) + " seconds)\n"
                f.writelines([comment] + lines)
    if not res:
        return None
    else:
        print("Found disjoint quorums!")
        model = s.get_model()
        q1 = get_quorum(fbas, 'A', model, quorum_vars_inverse)
        q2 = get_quorum(fbas, 'B', model, quorum_vars_inverse)
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        assert fbas.is_quorum(q1) and fbas.is_quorum(q2) and not set(q1) & set(q2)
        return (q1, q2)
    
def maximize(wcnf:WCNF) -> Optional[Tuple[int, Any]]:
    if config.max_sat_algo == 'LRU':
        s = LSU(wcnf)
    else:
        s = RC2(wcnf)
    start_time = time.time()
    if config.max_sat_algo == 'LRU':
        res = s.solve()
    else:
        res = s.compute()
    end_time = time.time()
    logging.info("Solving time: %s", end_time - start_time)
    if res:
        return s.cost, s.model
    return None

def make_faulty_vars(fbas) -> Tuple[dict[str,int], dict[int,str]]:
    """
    Create variables indicating whether a validator is faulty
    """
    global next_int
    is_faulty_vars = {}
    is_faulty_vars_inverse = {}
    for v in fbas.validators:
        is_faulty_vars[v] = next_int
        is_faulty_vars_inverse[next_int] = v
        next_int += 1
    return (is_faulty_vars, is_faulty_vars_inverse)
    
def find_minimal_splitting_set(fbas: FBASGraph) ->  Optional[Tuple[Collection,Collection,Collection]]:
    """
    Find a minimal-cardinality splitting set in the FBAS graph, or prove there is none.
    Uses one of pysat's MaxSAT procedures (LRU or RC2).
    If found, returns the splitting set and the two quorums that it splits.
    """

    if config.heuristic_first:
        logging.info("Computing lower bound on the splitting-set size using a fast heuristic (usually too conservative on non-symmetric networks)")
        res = fbas.splitting_set_bound()
        print(f"Lower bound on the splitting-set size: {res}")

    logging.info("Finding minimal-cardinality splitting set using MaxSAT algorithm %s with %s cardinality encoding", config.max_sat_algo, config.card_encoding)

    start_time = time.time()

    # the clauses of the CNF formula:
    clauses: Clauses = []

    # first, for each vertex in the FBAS graph, create two variables each indicating whether the vertex is in quorum A or quorum B.
    # also create a map to keep track of which variables encodes what.
    quorum_vars = {}
    quorum_vars_inverse = {}
    for q in ['A', 'B']:
        vars_, inverse = make_quorum_vars(q, fbas)
        quorum_vars.update(vars_)
        quorum_vars_inverse.update(inverse)

    # create variables indicating whether a validator is faulty:
    is_faulty_vars, is_faulty_vars_inverse = make_faulty_vars(fbas)

    # now we create the constraints:
    for q in ['A', 'B']: # for each of our two quorums
        # first, we create clauses asserting that the quorum contains at least one non-faulty validator:
        terms = [[quorum_vars[(q, v)], -is_faulty_vars[v]] for v in fbas.validators]
        clauses += dnf_to_cnf(terms)
        # then, we add the threshold constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [quorum_vars[(q, n)] for n in fbas.graph.successors(v)]
                if v in fbas.validators:
                    # the threshold must be met only if the validator is not faulty:
                    clauses += card_constraint_to_cnf([quorum_vars[(q, v)], -is_faulty_vars[v]], vs, fbas.threshold(v))
                else:
                    # the threshold must be met:
                    clauses += card_constraint_to_cnf([quorum_vars[(q, v)]], vs, fbas.threshold(v))
            if fbas.threshold(v) == 0:
                continue # no constraints for this vertex
            if fbas.threshold(v) < 0: # validators for which we don't have a threshold cannot be in the quorum:
                clauses.append([-quorum_vars[(q, v)]])
    # add the constraint that no non-faulty validator can be in both quorums:
    for v in fbas.validators:
        clauses.append([is_faulty_vars[v], -quorum_vars[('A', v)], -quorum_vars[('B', v)]])
    # finally, convert to weighted CNF and add soft constraints that minimize the number of faulty validators:
    wcnf = WCNF()
    wcnf.extend(clauses)
    for v in fbas.validators:
        wcnf.append([-is_faulty_vars[v]], weight=1)

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
        ss = [is_faulty_vars_inverse[i] for i in model if i in is_faulty_vars_inverse.keys()]
        logging.info("Minimal-cardinality splitting set: %s", [fbas.with_name(s) for s in ss])
        q1 = get_quorum(fbas,'A', model, quorum_vars_inverse)
        q2 = get_quorum(fbas, 'B',  model, quorum_vars_inverse)
        logging.info("Quorum A: %s", [fbas.with_name(v) for v in q1])
        logging.info("Quorum B: %s", [fbas.with_name(v) for v in q2])
        return (ss, q1, q2)

def find_minimal_blocking_set(fbas: FBASGraph) -> Optional[Collection[str]]:
    """
    Find a minimal-cardinality blocking set in the FBAS graph, or prove there is none.
    """
    raise NotImplementedError("Not implemented yet")
    logging.info("Finding minimal-cardinality blocking set using MaxSAT algorithm %s with %s cardinality encoding", config.max_sat_algo, config.card_encoding)

    start_time = time.time()

    # the clauses of the CNF formula:
    clauses: Clauses = []

    max_depth = fbas.max_depth()

    global next_int
    # for each vertex, create a variable indicating whether it's blocked at level i (0 <= i <= max_depth), and the assiciated constraints:
    blocked_vars = {}
    blocked_vars_inverse = {}
    for v in fbas.vertices():
        blocked_vars[(v,0)] = next_int
        blocked_vars_inverse[next_int] = (v,0)
        next_int += 1
    for i in range(1,max_depth+1):
        for v in fbas.vertices():
            blocked_vars[(v,i)] = next_int
            blocked_vars_inverse[next_int] = (v,i)
            next_int += 1
        # create the constraints:
        for v in fbas.vertices():
            if fbas.threshold(v) > 0:
                vs = [blocked_vars[(n,i-1)] for n in fbas.graph.successors(v)]
                clauses += [] # TODO
    return None

def find_disjoint_quorums_using_pysat_fmla(fbas: FBASGraph) -> Optional[Tuple[Collection, Collection]]:
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
    s = Solver(bootstrap_with=clauses, name=config.sat_solver)
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


def find_disjoint_quorums_(fbas: FBASGraph) -> Optional[Tuple[Collection, Collection]]:
    """
    Encodes the problem in propositional logic, apply the Tseitin transformation to convert to CNF,
    and then calls a SAT solver.
    """
    logging.info("Finding disjoint quorums by encoding to propositional logic. Using %s cardinality encoding and solver %s", config.card_encoding, config.sat_solver)

    def in_quorum(q:str, n:str):
        return pl.Atom((q, n))

    def get_quorum_from_atoms(atoms:list[int], q:str) -> list[str]:
        """Given a list of SAT atoms, return the validators in quorum q."""
        return [pl.variables_inv[v][1] for v in pl.variables.values() \
                if v in atoms and pl.variables_inv[v][0] == q and pl.variables_inv[v][1] in fbas.validators]
    
    def all_in_quorum(s: Collection, q : str) -> pl.Formula:
        return pl.And(*[in_quorum(q, n) for n in s])
    
    def quorum_satisfies_requirements_of(n: str, q: str) -> pl.Formula:
        if fbas.threshold(n) > 0:
            return pl.Or(*[all_in_quorum(s, q)
                for s in combinations(fbas.graph.successors(n), fbas.threshold(n))])
        elif fbas.threshold(n) == 0:
            return pl.And()
        else:
            return pl.Or()

    start_time = time.time()
    constraints : list[pl.Formula] = []
    for q in ['A', 'B']: # our two quorums
        # the quorum must be non-empty:
        constraints += [pl.Or(*[in_quorum(q, n) for n in fbas.validators])]
        # the quorum must satisfy the requirements of each of its members:
        constraints += \
            [pl.Implies(in_quorum(q, n), quorum_satisfies_requirements_of(n, q))
                for n in fbas.graph.nodes()]
    # no validator can be in both quorums:
    for v in fbas.validators:
        constraints += [pl.Not(pl.And(in_quorum('A', v), in_quorum('B', v)))]
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
    if res:
        model = s.get_model()
        q1 = get_quorum_from_atoms(model, 'A')
        q2 = get_quorum_from_atoms(model, 'B')
        logging.info("Disjoint quorums found")
        logging.info("Quorum A: %s", q1)
        logging.info("Quorum B: %s", q2)
        assert fbas.is_quorum(q1)
        assert fbas.is_quorum(q2)
        assert not set(q1) & set(q2)
        return (q1, q2)
    return None