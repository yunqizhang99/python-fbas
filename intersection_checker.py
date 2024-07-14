from itertools import combinations
from pysat.solvers import Solver
from pysat.card import *
from pysat.formula import *
from pysat.examples.lsu import LSU
from pysat.examples.optux import OptUx

from fbas import QSet, FBAS

def to_cnf(fmlas : list[Formula]):
    return [c for f in fmlas for c in f]

def intersection_constraints(fbas : FBAS):

    """
    Computes a formula that is satisfiable if and only if there exist two disjoint quorums in the FBAS.
    The idea is to create two propositional variables ('A',v) and ('B',v) for each validator v.
    ('A',v) indicates whether v is in a quorum 'A', and ('B',v) indicates whether v is in a quorum 'B'.
    Then we create constraints asserting that 'A' is a non-empty quorum and 'B' is a non-empty quorum.
    Finally we assert that no validator is in both quorums, and we check for satisfiability.
    If the constraints are satisfiable, then we have two disjoint quorums.
    Otherwise, we know that no two disjoint quorums exist.
    """
    constraints : list[Formula] = []
    def in_quorum(q, x):
        return Atom((q, x))
    for q in ['A', 'B']:
        # q is not empty:
        constraints += [Or(*[in_quorum(q,v) for v in fbas.qset_map.keys()])]
        # each member must have a slice in the quorum:
        def qset_satisfied(qs : QSet):
            slices = list(combinations(qs.validators | qs.inner_qsets, qs.threshold))
            return Or(*[And(*[in_quorum(q,x) for x in s]) for s in slices])
        constraints += [Implies(in_quorum(q,v), in_quorum(q, fbas.qset_map[v]))
                        for v in fbas.qset_map.keys()]
        constraints += [Implies(in_quorum(q, qs), qset_satisfied(qs))
                        for qs in fbas.all_qsets()]
    # no validator can be in both quorums:
    for v in fbas.qset_map.keys():
        constraints += [Neg(And(in_quorum('A', v), in_quorum('B', v)))]
    # convert to CNF and return:
    return to_cnf(constraints)

def get_quorum_from_formulas(fmlas, q):
    return [a.object[1] for a in fmlas
            if isinstance(a, Atom) and a.object[0] == q and not isinstance(a.object[1], QSet)]
    
def check_intersection(fbas : FBAS, solver='cms'):
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

def min_splitting_set_constraints(fbas : FBAS) -> WCNF:
    """
    Returns a formula that encdes the problem of finding a set of nodes that, if malicious, can cause two quorums to intersect only at malicious nodes, and that is of minimal cardinality.
    The formula consists of a set of hard constraints plus a set of soft constraints.
    A maxSAT solver will try to satisfy all the hard constraints and as many soft constraints as possible.

    The idea of the encoding is like for intersection checking, except that for each validator we add one variable indicating malicious failure, and we adjust the constraints accordingly.
    For each validator, we add a soft constraint stating that it is not malicious.
    """
    constraints : list[Formula] = []
    def in_quorum(q, x):
        return Atom((q, x))
    def malicious(v):
        return Atom(('malicious',v))
    for q in ['A', 'B']:
        # q has at least one non-faulty member:
        constraints += [Or(*[And(in_quorum(q,v), Neg(malicious(v))) for v in fbas.qset_map.keys()])]
        # each member must have a slice in the quorum, unless it's faulty:
        def qset_constraint(qs : QSet):
            slices = list(combinations(qs.validators | qs.inner_qsets, qs.threshold))
            return Implies(in_quorum(q, qs), Or(*[And(*[in_quorum(q, x) for x in s]) for s in slices]))
        constraints += [qset_constraint(qs) for qs in fbas.all_qsets()]
        constraints += [Implies(And(in_quorum(q, v), Neg(malicious(v))), in_quorum(q, fbas.qset_map[v]))
                        for v in fbas.qset_map.keys()]
    # no non-malicious validator can be in both quorums:
    for v in fbas.qset_map.keys():
        constraints += [Neg(And(Neg(malicious(v)), in_quorum('A', v), in_quorum('B', v)))]
    # convert to weighted CNF, which allows us to add soft constraints to be maximized:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    # add soft constraints for minimizing the number of malicious nodes (i.e. maximizing the number of non-malicious nodes):
    for v in fbas.qset_map.keys():
        nm = Neg(malicious(v))
        nm.clausify()
        wcnf.append(to_cnf([nm])[0], weight=1)
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
        malicious_nodes = [a.object[1] for a in fmlas if isinstance(a, Atom) and a.object[0] == 'malicious']
        print("Malicious nodes:", malicious_nodes)
        return malicious_nodes
    else:
        return frozenset()
    
def min_blocking_set_mus_constraints(fbas : FBAS) -> WCNF:
    """
    
    Assert that there is a quorum of non-failed validators, and for each validator add a soft clause asserting that it failed. The ask for an MUS of minimal size.
    """
    constraints : list[Formula] = []
    def in_quorum(x):
        return Atom(('in_quorum', x))
    def failed(v) -> Formula:
        return Atom(('failed', v))
    # the quorum contains only non-failed validators:
    constraints += [Implies(in_quorum(v), Neg(failed(v))) for v in fbas.qset_map.keys()]
    # the quorum is non-empty:
    constraints += [Or(*[in_quorum(v) for v in fbas.qset_map.keys()])]
    # each member must have a slice in the quorum, unless it's failed:
    def qset_constraint(qs : QSet):
        slices = list(combinations(qs.validators | qs.inner_qsets, qs.threshold))
        return Implies(in_quorum(qs), Or(*[And(*[in_quorum(x) for x in s]) for s in slices]))
    constraints += [qset_constraint(qs) for qs in fbas.all_qsets()]
    constraints += [Implies(in_quorum(v), in_quorum(fbas.qset_map[v]))
                        for v in fbas.qset_map.keys()]
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    for v in fbas.qset_map.keys():
        f = failed(v)
        f.clausify()
        wcnf.append(to_cnf([f])[0], weight=1)
    return wcnf
    
def min_blocking_set_constraints(fbas : FBAS) -> WCNF:
    """
    For each validator, we create a variable indicating it's in the closure of failed validators and another variable indicating whether it's failed.
    We assert that all non-failed validators are in the closure of failed validators.
    We minimize the number of failed validators.
    TODO: We need a well-foundedness condition. This sounds difficult to encode. We could try to superpose a directed graph (with variables for edges), assert that it's acyclic, and make sure a node can only be blocked by predecessors.
    Another idea is to use minimal unsat set. That seems easier: assert that there is a quorum of non-failed validators, and for each validator add a soft clause asserting that it failed. The ask for a MUS. But that's minimal, not minimum.
    """
    constraints : list[Formula] = []
    def failed(v) -> Formula:
        return Atom(('failed', v))
    def in_closure(x) -> Formula:
        return Atom(('closure', x))
    # a validator is in the closure if and only if it has not failed:
    for v in fbas.qset_map.keys():
        constraints += [Equals(Neg(failed(v)), in_closure(v))]
    # a qset is in the closure if and only if every one of its slices has a member in the closure or failed:
    def qset_constraint(qs : QSet) -> Formula:
        blocking_threshold = len(qs.elements()) - qs.threshold + 1
        blocking = list(combinations(qs.validators | qs.inner_qsets, blocking_threshold))
        return Equals(
            Or(*[And(*[Or(in_closure(x), failed(x)) if x in qs.validators else in_closure(x)
                        for x in b]) for b in blocking]),
            in_closure(qs))
    constraints += [qset_constraint(qs) for qs in fbas.all_qsets()]
    # a validator is in the closure if and only if: its qset is or it failed:
    constraints += [Equals(Or(in_closure(fbas.qset_map[v]), failed(v)), in_closure(v))
                    for v in fbas.qset_map.keys()]
    # convert to weighted CNF, which allows us to add soft constraints to be maximized:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    # add soft constraints for minimizing the number of failed nodes (i.e. maximizing the number of non-failed nodes):
    for v in fbas.qset_map.keys():
        nf = Neg(failed(v))
        nf.clausify()
        wcnf.append(to_cnf([nf])[0], weight=1)
    raise NotImplementedError("This function is not yet implemented")
    return wcnf

def min_blocking_set(fbas):
    """Returns a blocking set of minimum cardinality"""
    wcnf = min_blocking_set_mus_constraints(fbas)
    # this is much faster but not minimum:
    # indices = OptUx(wcnf, puresat='mgh', unsorted=True).compute()
    indices = OptUx(wcnf).compute() # list of indices of soft clauses that are satisfied
    clauses = [wcnf.soft[i-1] for i in indices] # TODO: this -1 is very fishy
    failed = [f.object[1] for f in Formula.formulas([f for clause in clauses for f in clause])]
    print(f'Found minimal blocking set of size {len(failed)}: {failed}')
    return failed
