from itertools import combinations
from pysat.solvers import Solver
from pysat.card import *
from pysat.formula import *
from pysat.examples.lsu import LSU

from fbas import QSet

def to_cnf(fmlas):
    return [c for f in fmlas for c in f]

def intersection_constraints(fbas):

    """
    Computes a formula that is satisfiable if and only if there exist two disjoint quorums in the FBAS.
    The idea is to create two propositional variables ('A',v) and ('B',v) for each validator v.
    ('A',v) indicates whether v is in a quorum 'A', and ('B',v) indicates whether v is in a quorum 'B'.
    Then we create constraints asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for satisfiability.
    If the constraints are satisfiable, then we have two disjoint quorums.
    Otherwise, we know that no two disjoint quorums exist.
    """
    constraints = []
    for q in ['A', 'B']:
        # q is not empty:
        constraints += [Or(*[Atom((q,v)) for v in fbas.qset_map.keys()])]
        # each member must have a slice in the quorum:

        def qset_satisfied(qs):
            slices = list(combinations(qs.validators | qs.inner_qsets, qs.threshold))
            return Or(*[And(*[Atom((q, x)) for x in s]) for s in slices])
        constraints += [Implies(Atom((q, v)), Atom((q, fbas.qset_map[v])))
                        for v in fbas.qset_map.keys()]
        constraints += [Implies(Atom((q, qs)), qset_satisfied(qs))
                        for qs in fbas.all_qsets()]
    # no validator can be in both quorums:
    for v in fbas.qset_map.keys():
        constraints += [Neg(And(Atom(('A', v)), Atom(('B', v))))]
    # convert to CNF and return:
    return to_cnf(constraints)

def get_quorum_from_formulas(fmlas, q):
    return [a.object[1] for a in fmlas
            if isinstance(a, Atom) and a.object[0] == q and not isinstance(a.object[1], QSet)]
    
def check_intersection(fbas, solver='cms'):
    """Returns True if and only if all quorums intersect"""
    collapsed_fbas = fbas.collapse_qsets()
    clauses = intersection_constraints(collapsed_fbas)
    s = Solver(bootstrap_with=clauses, name=solver)
    res = s.solve()
    if res:
        model = s.get_model()
        fmlas = [f for f in Formula.formulas(model, atoms_only=True)]
        print("Disjoint quorums:")
        print("Quorum A:", get_quorum_from_formulas(fmlas, 'A'))
        print("Quorum B:", get_quorum_from_formulas(fmlas, 'B'))
    return not res

def min_splitting_set_constraints(fbas):
    """
    Returns a formula that encdes the problem of finding a set of nodes that, if malicious, can cause two quorums to intersect only at malicious nodes, and that is of minimal cardinality.
    The formula consists of a set of hard constraints plus a set of soft constraints.
    A maxSAT solver will try to satisfy all the hard constraints and as many soft constraints as possible.

    The idea of the encoding is like for intersection checking, except that for each validator we add one variable indicating malicious failure, and we adjust the constraints accordingly.
    For each validator, we add a soft constraint stating that it has not failed.
    """
    constraints = []
    def not_failed(v):
        return Neg(Atom(('failed',v)))
    for q in ['A', 'B']:
        # q has at least one non-faulty member:
        constraints += [Or(*[And(Atom((q,v)), not_failed(v)) for v in fbas.qset_map.keys()])]
        # each member must have a slice in the quorum, unless it's faulty:
        def qset_satisfied(qs):
            slices = list(combinations(qs.validators | qs.inner_qsets, qs.threshold))
            return Or(*[And(*[Atom((q, x)) for x in s]) for s in slices])
        constraints += [Implies(And(Atom((q, v)), not_failed(v)), Atom((q, fbas.qset_map[v])))
                        for v in fbas.qset_map.keys()] 
        constraints += [Implies(Atom((q, qs)), qset_satisfied(qs))
                        for qs in fbas.all_qsets()]
    # no non-failed validator can be in both quorums:
    for v in fbas.qset_map.keys():
        constraints += [Neg(And(not_failed(v), Atom(('A', v)), Atom(('B', v))))]
    # convert to weighted CNF, which allows us to add soft constraints:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    # add soft constraints for minimizing the number of failed nodes (i.e. maximizing the number of non-failed nodes):
    for v in fbas.qset_map.keys():
        nf = not_failed(v)
        nf.clausify()
        wcnf.append(to_cnf([nf])[0], weight=1)
    return wcnf

def min_splitting_set(fbas, solver_class=LSU):  # LSU seems to perform the best
    """Returns a splitting set of minimum cardinality"""
    wncf = min_splitting_set_constraints(fbas)
    maxSAT_solver = solver_class(wncf)
    got_model = maxSAT_solver.compute() if 'compute' in solver_class.__dict__ else maxSAT_solver.solve()
    if got_model:
        print(f'Found minimal splitting set of size {maxSAT_solver.cost}')
        model = maxSAT_solver.model
        fmlas = [f for f in Formula.formulas(model, atoms_only=True)]
        print("Disjoint quorums:")
        print("Quorum A:", get_quorum_from_formulas(fmlas, 'A'))
        print("Quorum B:", get_quorum_from_formulas(fmlas, 'B'))
        failed_nodes = [a.object[1] for a in fmlas if isinstance(a, Atom) and a.object[0] == 'failed']
        print("Failed nodes:", failed_nodes)
        return failed_nodes
    else:
        return frozenset()