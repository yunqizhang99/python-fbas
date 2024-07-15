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
    We also need a well-foundedness condition. For this we superpose a directed graph (with variables for edges), assert that it's acyclic, and make sure a node can only be blocked by predecessors.
    It works on small examples but is otherwise extremely slow.
    """
    constraints : list[Formula] = []
    def failed(v) -> Formula:
        return Atom(('failed', v))
    def in_closure(x) -> Formula:
        return Atom(('closure', x))
    def edge(x, y) -> Formula:
        return Atom(('graph', x, y))
    # the edge relation is transitive:
    # TODO: just this seems too big to construct
    constraints += [Implies(And(edge(x,y), edge(y,z)), edge(x,z))
                    for x in fbas.elements() for y in fbas.elements() for z in fbas.elements()]
    # acyclicity:
    constraints += [Neg(edge(x,x)) for x in fbas.elements()]
    # no edge to failed validators:
    constraints += [Neg(And(edge(x,y), failed(y))) for x in fbas.qset_map.keys() for y in fbas.qset_map.keys()]
    # a non-failed validator is in the closure if and only if its qset is in it and is a predecessor in the graph:
    constraints += [Equals(
        And(in_closure(v), Neg(failed(v))),
        And(edge(fbas.qset_map[v],v), in_closure(fbas.qset_map[v]))) for v in fbas.qset_map.keys()]
    # a qset is in the closure if and only if it's blocked by members of the closure that are predecessors in the graph:
    def qset_in_closure(qs: QSet):
        blocking_threshold = len(qs.elements()) - qs.threshold + 1
        return Or(*[And(*[And(in_closure(x), edge(x,qs)) for x in s]) for s in combinations(qs.elements(), blocking_threshold)])
    constraints += [Equals(qset_in_closure(qs), in_closure(qs)) for qs in fbas.all_qsets()]
    # the closure contains all validators and qsets::
    constraints += [in_closure(x) for x in fbas.qset_map.keys() | fbas.all_qsets()]
    # finally, we want to maximize the number of non-failed validators:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    for v in fbas.qset_map.keys():
        f = Neg(failed(v))
        f.clausify()
        wcnf.append(to_cnf([f])[0], weight=1)
    return wcnf

def min_blocking_set(fbas, solver_class=LSU):  # LSU seems to perform the best
    """Returns a splitting set of minimum cardinality"""
    wncf = min_blocking_set_constraints(fbas)
    maxSAT_solver = solver_class(wncf)
    got_model = maxSAT_solver.compute() if 'compute' in solver_class.__dict__ else maxSAT_solver.solve()
    if got_model:
        print(f'Found minimal blocking set of size {maxSAT_solver.cost}')
        model = maxSAT_solver.model
        fmlas = [f for f in Formula.formulas(model, atoms_only=True)]
        failed_nodes = [a.object[1] for a in fmlas if isinstance(a, Atom) and a.object[0] == 'failed']
        print("Failed nodes:", failed_nodes)
        return failed_nodes
    else:
        return frozenset()
    
def min_blocking_set_mus(fbas):
    """Returns a blocking set of minimum cardinality"""
    wcnf = min_blocking_set_mus_constraints(fbas)
    def get_failed(indices):
        clauses = [wcnf.soft[i-1] for i in indices]
        return [f.object[1] for f in Formula.formulas([f for clause in clauses for f in clause])]
    # this is much faster but a minimal MUS, but not necessarily a minimum-cardinality one:
    # indices = OptUx(wcnf, puresat='mgh', unsorted=True).compute()
    indices = OptUx(wcnf).compute() # list of indices of soft clauses that are satisfied
    failed = get_failed(indices)
    # with OptUx(wcnf, puresat='mgh', unsorted=True) as optux:
        # for mus in optux.enumerate():
            # print(f'Found minimal blocking set of size {len(mus)}: {get_failed(mus)}')
    print(f'Found minimal blocking set of size {len(failed)}: {failed}')
    return failed
