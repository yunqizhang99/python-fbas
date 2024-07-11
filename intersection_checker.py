from fbas import *
from pysat.card import *
from pysat.formula import *
from pysat.solvers import Solver
from itertools import combinations

def check_intersection(fbas):

    """
    Checks whether all quorums intersect.
    The idea is to create two propositional variables ('A',v) and ('B',v) for each validator v.
    ('A',v) indicates whether v is in a quorum 'A', and ('B',v) indicates whether v is in a quorum 'B'.
    Then we create constraints asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for satisfiability.
    If the constraints are satisfiable, then we have two disjoint quorums.
    Otherwise, we know that no two disjoint quorums exist.
    """
    constraints = []  # cs will hold the constraints
    for q in ['A', 'B']:
        # q is not empty:
        constraints += [Or(*[Atom((q,v)) for v in fbas.qset_map.keys()])]
        # each member must have a slice in the quorum:

        def qset_satisfied(qs):
            slices = list(combinations(qs.validators | qs.innerQSets, qs.threshold))
            return Or(*[And(*[Atom((q, x)) for x in s]) for s in slices])
        constraints += [Implies(Atom((q, v)), qset_satisfied(fbas.qset_map[v])) for v in fbas.qset_map.keys()]
        constraints += [Implies(Atom((q, qs)), qset_satisfied(qs)) for qs in fbas.all_QSets()]
    # no validator can be in both quorums:
    for v in fbas.qset_map.keys():
        constraints += [Neg(And(Atom(('A', v)), Atom(('B', v))))]

    # convert all the constraints to clauses:
    big_conj = And(*constraints)  # needed?
    clauses = [c for c in big_conj]
    # solve:
    s = Solver(bootstrap_with=clauses)
    res = s.solve()
    # debug:
    if res:
        model = s.get_model()
        print(model)
        print(Formula.formulas(model, atoms_only=True))
    return not res
