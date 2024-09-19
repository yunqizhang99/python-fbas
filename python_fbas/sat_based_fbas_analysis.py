from pysat.solvers import Solver
from pysat.card import *
from pysat.formula import *
from pysat.examples.lsu import LSU
from pysat.examples.optux import OptUx

from .fbas import QSet, FBAS

def _to_cnf(fmlas : list[Formula]):
    return [c for f in fmlas for c in f]

def _cnf_of_pseudo_atom(a: Formula) -> list:
    a.clausify()
    return _to_cnf([a])[0]

def _intersection_constraints(fbas : FBAS):

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
        constraints += [Or(*[in_quorum(q,v) for v in fbas.validators()])]
        # each member must have a slice in the quorum:
        def qset_satisfied(qs : QSet):
            return Or(*[And(*[in_quorum(q,x) for x in s]) for s in qs.slices()])
        constraints += [Implies(in_quorum(q,v), in_quorum(q, fbas.qset_map[v]))
                        for v in fbas.validators()]
        constraints += [Implies(in_quorum(q, qs), qset_satisfied(qs))
                        for qs in fbas.all_qsets()]
    # no validator can be in both quorums:
    for v in fbas.validators():
        constraints += [Neg(And(in_quorum('A', v), in_quorum('B', v)))]
    # convert to CNF and return:
    return _to_cnf(constraints)

# TODO: rename to get_tagged_validators?
def _get_quorum_from_formulas(fmlas, q):
    return [a.object[1] for a in fmlas
            if isinstance(a, Atom) and a.object[0] == q and not isinstance(a.object[1], QSet)]
    
def check_intersection(fbas : FBAS, solver='cms'):
    """Returns True if and only if all quorums intersect"""
    collapsed_fbas = fbas.collapse_qsets()
    clauses = _intersection_constraints(collapsed_fbas)
    s = Solver(bootstrap_with=clauses, name=solver)
    res = s.solve()
    if res:
        model = s.get_model()
        fmlas = [f for f in Formula.formulas(model, atoms_only=True)]
        print("Disjoint quorums:")
        print("Quorum A:", _get_quorum_from_formulas(fmlas, 'A'))
        print("Quorum B:", _get_quorum_from_formulas(fmlas, 'B'))
    return not res

def _min_splitting_set_constraints(fbas : FBAS, labels = None) -> WCNF:
    """
    Returns a formula that encodes the problem of finding a set of nodes that, if malicious, can cause two quorums to intersect only at malicious nodes, and that is of minimal cardinality.
    The formula consists of a set of hard constraints plus a set of soft constraints.
    A maxSAT solver will try to satisfy all the hard constraints and as many soft constraints as possible.

    The idea of the encoding is like for intersection checking, except that for each validator we add one variable indicating malicious failure, and we adjust the constraints accordingly.
    For each validator, we add a soft constraint stating that it is not malicious.

    TODO: if labels is not None, minimize the number of labels instead of the number of validators.
    """
    constraints : list[Formula] = []
    def in_quorum(q, x):
        return Atom((q, x))
    def malicious(v):
        return Atom(('malicious',v))
    for q in ['A', 'B']:
        # q has at least one non-faulty member:
        constraints += [Or(*[And(in_quorum(q,v), Neg(malicious(v))) for v in fbas.validators()])]
        # each member must have a slice in the quorum, unless it's faulty:
        def qset_constraint(qs : QSet):
            return Implies(in_quorum(q, qs), Or(*[And(*[in_quorum(q, x) for x in s]) for s in qs.slices()]))
        constraints += [qset_constraint(qs) for qs in fbas.all_qsets()]
        constraints += [Implies(And(in_quorum(q, v), Neg(malicious(v))), in_quorum(q, fbas.qset_map[v]))
                        for v in fbas.validators()]
    # no non-malicious validator can be in both quorums:
    for v in fbas.validators():
        constraints += [Neg(And(Neg(malicious(v)), in_quorum('A', v), in_quorum('B', v)))]
    # convert to weighted CNF, which allows us to add soft constraints to be maximized:
    wcnf = WCNF()
    wcnf.extend(_to_cnf(constraints))
    # add soft constraints for minimizing the number of malicious nodes (i.e. maximizing the number of non-malicious nodes):
    for v in fbas.validators():
        nm = Neg(malicious(v))
        wcnf.append(_cnf_of_pseudo_atom(nm), weight=1)
    return wcnf

def min_splitting_set(fbas, solver_class=LSU):  # LSU seems to perform the best
    """Returns a splitting set of minimum cardinality"""
    wncf = _min_splitting_set_constraints(fbas)
    maxSAT_solver = solver_class(wncf)
    got_model = maxSAT_solver.compute() if 'compute' in solver_class.__dict__ else maxSAT_solver.solve()
    if got_model:
        print(f'Found minimal splitting set of size {maxSAT_solver.cost}')
        model = maxSAT_solver.model
        fmlas = [f for f in Formula.formulas(model, atoms_only=True)]
        print("Disjoint quorums:")
        print("Quorum A:", _get_quorum_from_formulas(fmlas, 'A'))
        print("Quorum B:", _get_quorum_from_formulas(fmlas, 'B'))
        malicious_nodes = [a.object[1] for a in fmlas if isinstance(a, Atom) and a.object[0] == 'malicious']
        print("Malicious nodes:", malicious_nodes)
        return malicious_nodes
    else:
        return frozenset()
    
def _min_blocking_set_mus_constraints(fbas : FBAS) -> WCNF:
    """
    
    Assert that there is a quorum of non-failed validators, and for each validator add a soft clause asserting that it failed. Then ask for an MUS of minimal size.
    """
    constraints : list[Formula] = []
    def in_quorum(x):
        return Atom(('in_quorum', x))
    def failed(v) -> Formula:
        return Atom(('failed', v))
    # the quorum contains only non-failed validators:
    constraints += [Implies(in_quorum(v), Neg(failed(v))) for v in fbas.validators()]
    # the quorum is non-empty:
    constraints += [Or(*[in_quorum(v) for v in fbas.validators()])]
    # each member must have a slice in the quorum, unless it's failed:
    def qset_constraint(qs : QSet):
        return Implies(in_quorum(qs), Or(*[And(*[in_quorum(x) for x in s]) for s in qs.slices()]))
    constraints += [qset_constraint(qs) for qs in fbas.all_qsets()]
    constraints += [Implies(in_quorum(v), in_quorum(fbas.qset_map[v]))
                        for v in fbas.validators()]
    wcnf = WCNF()
    wcnf.extend(_to_cnf(constraints))
    for v in fbas.validators():
        f = failed(v)
        wcnf.append(_cnf_of_pseudo_atom(f), weight=1)
    return wcnf
        
def min_blocking_set_mus(fbas):
    """Returns a blocking set of minimum cardinality"""
    wcnf = _min_blocking_set_mus_constraints(fbas)
    def get_failed(indices):
        clauses = [wcnf.soft[i-1] for i in indices]
        return [f.object[1] for f in Formula.formulas([f for clause in clauses for f in clause])]
    # this is much faster, but does not necessarily return a minimum-cardinality MUS (the returned unsatisfiable subset is minimal, but not necessarily of smallest cardinality):
    # indices = OptUx(wcnf, puresat='mgh', unsorted=True).compute()
    indices = OptUx(wcnf).compute() # list of indices of soft clauses that are satisfied
    failed = get_failed(indices)
    print(f'Found minimal blocking set of size {len(failed)}: {failed}')
    return failed

def _min_blocking_set_constraints(fbas : FBAS) -> WCNF:
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
    constraints += [Neg(And(edge(x,y), failed(y))) for x in fbas.validators() for y in fbas.validators()]
    # a non-failed validator is in the closure if and only if its qset is in it and is a predecessor in the graph:
    constraints += [Equals(
        And(in_closure(v), Neg(failed(v))),
        And(edge(fbas.qset_map[v],v), in_closure(fbas.qset_map[v]))) for v in fbas.validators()]
    # a qset is in the closure if and only if it's blocked by members of the closure that are predecessors in the graph:
    def qset_in_closure(qs: QSet):
        return Or(*[And(*[And(in_closure(x), edge(x,qs)) for x in s]) for s in qs.v_blocking_sets()])
    constraints += [Equals(qset_in_closure(qs), in_closure(qs)) for qs in fbas.all_qsets()]
    # the closure contains all validators and qsets::
    constraints += [in_closure(x) for x in fbas.validators() | fbas.all_qsets()]
    # finally, we want to maximize the number of non-failed validators:
    wcnf = WCNF()
    wcnf.extend(_to_cnf(constraints))
    for v in fbas.validators():
        f = Neg(failed(v))
        wcnf.append(_cnf_of_pseudo_atom(f), weight=1)
    return wcnf

def min_blocking_set(fbas, solver_class=LSU):  # LSU seems to perform the best
    """Returns a splitting set of minimum cardinality"""
    wncf = _min_blocking_set_constraints(fbas)
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

def _optimal_overlay_constraints(fbas):
    # TODO: try to minimize max degree with an ILP
    # TODO: this is costly; do it only on the org hypergraph
    constraints : list[Formula] = []
    def edge(x, y) -> Formula:
        return Atom((x, y))
    # symmetric graph:
    constraints += [Implies(edge(x,y), edge(y,x)) for x in fbas.validators() for y in fbas.validators()]
    # no self edges:
    constraints += [Neg(edge(x,x)) for x in fbas.validators()]
    # diameter is at most 2:
    constraints += [Or(*[And(edge(x,z), edge(z,y)) for z in fbas.validators()]) for x in fbas.validators() for y in fbas.validators()]
    # each validator has at least one neighbor in each of its slices:
    # TODO: we need real slices, not stuff that has blocking sets in it. Or maybe we can eliminate them later.
    constraints += [Or(*[And(*[edge(v, neighbor) for neighbor in blocking]) for blocking in fbas.qset_map[v].v_blocking_sets()])
        for v in fbas.validators()]
    # minimize the number of edges:
    wcnf = WCNF()
    wcnf.extend(_to_cnf(constraints))
    for v1 in fbas.validators():
        for v2 in fbas.validators() - {v1}:
            f = Neg(edge(v1,v2))
            wcnf.append(_cnf_of_pseudo_atom(f), weight=1)
    return wcnf

def optimal_overlay(fbas, solver_class=LSU):  # LSU seems to perform the best
    wncf = _optimal_overlay_constraints(fbas)
    print("computed constraints")
    maxSAT_solver = solver_class(wncf)
    got_model = maxSAT_solver.compute() if 'compute' in solver_class.__dict__ else maxSAT_solver.solve()
    if got_model:
        print(f'Found minimal overlay of cost {maxSAT_solver.cost}')
        model = maxSAT_solver.model
        fmlas = [f for f in Formula.formulas(model, atoms_only=True)]
        edges = [a.object for a in fmlas if isinstance(a, Atom)]
        # remove symmetric edges:
        edges = [edge for edge in edges if edge[0] < edge[1]]
        print("edges:", edges)
        return frozenset(edges)
    else:
        return frozenset()