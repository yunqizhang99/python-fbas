from itertools import combinations
from pysat.solvers import Solver
from pysat.card import *
from pysat.formula import *

def intersection_constraints(fbas, solver='cms'):

    """
    Checks whether all quorums intersect.
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
        constraints += [Implies(Atom((q, v)), qset_satisfied(fbas.qset_map[v]))
                        for v in fbas.qset_map.keys()]
        constraints += [Implies(Atom((q, qs)), qset_satisfied(qs))
                        for qs in fbas.all_qsets()]
    # no validator can be in both quorums:
    for v in fbas.qset_map.keys():
        constraints += [Neg(And(Atom(('A', v)), Atom(('B', v))))]
    return constraints

def check_intersection(fbas, solver='cms'):
    clauses = [c for cstr in intersection_constraints(fbas, solver) for c in cstr]
    s = Solver(bootstrap_with=clauses, name=solver)
    res = s.solve()
    if res:
        model = s.get_model()
        def get_quorum(q):
            return [a.object[1] for a in Formula.formulas(model, atoms_only=True)
                    if isinstance(a, Atom) and a.object[0] == q]
        print("Disjoint quorums:")
        print("Quorum A:", get_quorum('A'))
        print("Quorum B:", get_quorum('B'))
    return not res