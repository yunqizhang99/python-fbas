import logging
from pysat.solvers import Solver
from pysat.card import *
from pysat.formula import *
from pysat.examples.lsu import LSU
from pysat.examples.optux import OptUx
from pyqbf.formula import PCNF
from pyqbf.solvers import Solver as QSolver

from python_fbas.deprecated.fbas import QSet, FBAS
from python_fbas.utils import to_cnf, clause_of_pseudo_atom, vars_of_cnf

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
        constraints += [Implies(in_quorum(q,v), in_quorum(q, fbas.qset_map[v]))
                        for v in fbas.validators()]
        def qset_satisfied(qs : QSet):
            return Or(*[And(*[in_quorum(q,x) for x in s]) for s in qs.level_1_slices()])
        constraints += [Implies(in_quorum(q, qs), qset_satisfied(qs))
                        for qs in fbas.all_qsets()]
    # no validator can be in both quorums:
    for v in fbas.validators():
        constraints += [Neg(And(in_quorum('A', v), in_quorum('B', v)))]
    # convert to CNF and return:
    return to_cnf(constraints)

# TODO: rename to get_tagged_validators?
def _get_quorum_from_atoms(fmlas, q):
    return [a.object[1] for a in fmlas
            if isinstance(a, Atom) and a.object[0] == q and not isinstance(a.object[1], QSet)]
    
def check_intersection(fbas : FBAS, solver='cms'):
    """Returns True if and only if all quorums intersect"""
    # TODO debug collapse_qsets (see test_empty_qset)
    # collapsed_fbas = fbas.collapse_qsets()
    collapsed_fbas = fbas
    clauses = _intersection_constraints(collapsed_fbas)
    s = Solver(bootstrap_with=clauses, name=solver)
    res = s.solve()
    if res:
        model = s.get_model()
        fmlas = [f for f in Formula.formulas(model, atoms_only=True)]
        print("Disjoint quorums:")
        print("Quorum A:", _get_quorum_from_atoms(fmlas, 'A'))
        print("Quorum B:", _get_quorum_from_atoms(fmlas, 'B'))
    return not res

def _min_splitting_set_constraints(fbas : FBAS, group_by = None) -> WCNF:
    """
    Returns a formula that encodes the problem of finding a set of nodes that, if malicious, can cause two quorums to intersect only at malicious nodes, and that is of minimal cardinality.
    The formula consists of a set of hard constraints plus a set of soft constraints.
    A maxSAT solver will try to satisfy all the hard constraints and as many soft constraints as possible.

    The idea of the encoding is like for intersection checking, except that for each validator we add one variable indicating malicious failure, and we adjust the constraints accordingly.
    For each validator, we add a soft constraint stating that it is not malicious.
    """

    # if group_by is not None, first check that all validators have a group:
    if group_by is not None and not fbas.all_have_meta_field(group_by):
        raise ValueError(f"Grouping field \"{group_by}\" is not defined for all validators")

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
            return Implies(in_quorum(q, qs), Or(*[And(*[in_quorum(q, x) for x in s]) for s in qs.level_1_slices()]))
        constraints += [qset_constraint(qs) for qs in fbas.all_qsets()]
        constraints += [Implies(And(in_quorum(q, v), Neg(malicious(v))), in_quorum(q, fbas.qset_map[v]))
                        for v in fbas.validators()]
    # no non-malicious validator can be in both quorums:
    for v in fbas.validators():
        constraints += [Neg(And(Neg(malicious(v)), in_quorum('A', v), in_quorum('B', v)))]
    # if group_by is not None, we add one variable per group denoting whether the group is malicious and we add constraints to ensure that all members of a group are malicious or none is:
    if group_by is not None:
        groups = fbas.meta_field_values(group_by)
        def group_members(g):
            return [v for v in fbas.validators() if fbas.metadata[v][group_by] == g]
        for g in groups:
            constraints += [Implies(malicious(g), And(*[malicious(v) for v in group_members(g)]))]
            constraints += [Implies(Or(*[malicious(v) for v in group_members(g)]), malicious(g))]
    # convert to weighted CNF, which allows us to add soft constraints to be maximized:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    if group_by is None:
        # add soft constraints for minimizing the number of malicious nodes (i.e. maximizing the number of non-malicious nodes):
        for v in fbas.validators():
            nm = Neg(malicious(v))
            wcnf.append(clause_of_pseudo_atom(nm), weight=1)
    else:
        # add soft constraints for minimizing the number of malicious groups:
        for g in groups:
            nm = Neg(malicious(g))
            wcnf.append(clause_of_pseudo_atom(nm), weight=1)
    return wcnf

def min_splitting_set(fbas, solver_class=LSU, group_by=None):  # LSU seems to perform the best
    """Returns a splitting set of minimum cardinality"""
    wcnf = _min_splitting_set_constraints(fbas, group_by)
    maxSAT_solver = solver_class(wcnf)
    got_model = maxSAT_solver.compute() if 'compute' in solver_class.__dict__ else maxSAT_solver.solve()
    if got_model:
        model = maxSAT_solver.model
        if group_by is not None:
            print(f'Found minimal splitting set of size {maxSAT_solver.cost} groups')
            fmlas = Formula.formulas(model, atoms_only=True)
            malicious_groups = [
                a.object[1] for a in fmlas
                    if isinstance(a, Atom) and a.object[0] == 'malicious'
                        and a.object[1] in fbas.meta_field_values(group_by)
            ]
            print("Minimal splitting set:", malicious_groups)
            logging.info("Disjoint quorums:")
            logging.info("Quorum A: %s", _get_quorum_from_atoms(fmlas, 'A'))
            logging.info("Quorum B: %s", _get_quorum_from_atoms(fmlas, 'B'))
            malicious_nodes = [
                a.object[1] for a in fmlas
                    if isinstance(a, Atom) and a.object[0] == 'malicious'
                        and a.object[1] not in fbas.meta_field_values(group_by)]
            logging.info("Splitting node set: %s", malicious_nodes)
            return malicious_groups
        else:
            print(f'Found minimal splitting set of size {maxSAT_solver.cost} nodes')
            fmlas = Formula.formulas(model, atoms_only=True)
            malicious_nodes = [a.object[1] for a in fmlas if isinstance(a, Atom) and a.object[0] == 'malicious']
            print("Minimal splitting set:", malicious_nodes)
            print("Quorums that intersect only in the splitting set:")
            print("Quorum A:", _get_quorum_from_atoms(fmlas, 'A'))
            print("Quorum B:", _get_quorum_from_atoms(fmlas, 'B'))
            return malicious_nodes
    else:
        print("No splitting set found")
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
        return Implies(in_quorum(qs), Or(*[And(*[in_quorum(x) for x in s]) for s in qs.level_1_slices()]))
    constraints += [qset_constraint(qs) for qs in fbas.all_qsets()]
    constraints += [Implies(in_quorum(v), in_quorum(fbas.qset_map[v]))
                        for v in fbas.validators()]
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    for v in fbas.validators():
        f = failed(v)
        wcnf.append(clause_of_pseudo_atom(f), weight=1)
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
        return Or(*[And(*[And(in_closure(x), edge(x,qs)) for x in s]) for s in qs.level_1_v_blocking_sets()])
    constraints += [Equals(qset_in_closure(qs), in_closure(qs)) for qs in fbas.all_qsets()]
    # the closure contains all validators and qsets::
    constraints += [in_closure(x) for x in fbas.validators() | fbas.all_qsets()]
    # finally, we want to maximize the number of non-failed validators:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    for v in fbas.validators():
        f = Neg(failed(v))
        wcnf.append(clause_of_pseudo_atom(f), weight=1)
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
    
def is_in_min_quorum_of(fbas : FBAS, v1, v2, solver='cms') -> bool:
    """
    Checks whether v1 is in a minimal quorum of v2 by checking the satisfiability of a quantified boolean formula.
    We assert that A is a quorum of v1 and v2, and that for all quorums B of v2, B is not a subset of A.
    If this is SAT, then v1 is in a minimal quorum of v2.

    One tricky part is that this has to be converted to a PCNF formula. We can do this using the Tseitin transformation but we need to make sure that we existentially quantify the new variables in the scope of the originally universally quantified variables. This formula should look like this: exists orginal_exists_vars. forall original_forall_vars. exists new_vars : CNF_formula
    """
    if v2 == v1:
        return True
    
    fbas = fbas.restrict_to_reachable(v2)
    # if v1 is not in the fbas anymore, return False:
    if v1 not in fbas.validators():
        return False
    
    A_constraints : list[Formula] = []
    B_constraints : list[Formula] = []
    def in_quorum(q, x):
        return Atom((q, x))
    def qset_satisfied(q : str, qs : QSet):
        return Or(*[And(*[in_quorum(q,x) for x in s]) for s in qs.level_1_slices()])
    
    A_constraints += [Implies(in_quorum('A',v), in_quorum('A', fbas.qset_map[v]))
                    for v in fbas.validators()]
    A_constraints += [Implies(in_quorum('A', qs), qset_satisfied('A', qs))
                    for qs in fbas.all_qsets()]
    A_constraints += [in_quorum('A', v2), in_quorum('A', v1)]

    B_constraints += [Implies(in_quorum('B',v), in_quorum('B', fbas.qset_map[v]))
                    for v in fbas.validators()]
    B_constraints += [Implies(in_quorum('B', qs), qset_satisfied('B', qs))
                    for qs in fbas.all_qsets()]
    B_constraints += [in_quorum('B', v2)]
    B_constraints += [Neg(And(in_quorum('B', v), Neg(in_quorum('A', v)))) for v in fbas.validators()]
    B_constraints += [Or(*[And(in_quorum('A',v), Neg(in_quorum('B',v))) for v in fbas.validators()])]

    A_clauses = to_cnf(A_constraints)
    B_clauses = to_cnf([Neg(And(*B_constraints))])
    A_lits : set[int] = vars_of_cnf(A_clauses)
    B_atoms = (
        [Atom(('B',v)) for v in fbas.validators()] +
            [Atom(('B',qs)) for qs in fbas.all_qsets()] )
    for b in B_atoms:
        b.clausify()
    B_atoms_lits = [b.clauses[0][0] for b in B_atoms]
    B_tseitin_lits : set[int] = vars_of_cnf(B_clauses) - A_lits - set(B_atoms_lits)

    pcnf = PCNF(from_clauses=A_clauses + B_clauses)
    pcnf.exists(*list(A_lits)).forall(*B_atoms_lits).exists(*list(B_tseitin_lits))

    # solvers: 'depqbf', 'qute', 'rareqs', 'qfun', 'caqe'
    s = QSolver(name='depqbf', bootstrap_with=pcnf)
    res = s.solve()
    if res:
        logging.info("%s is in a minimal quorum of %s", v1, v2)
        model = s.get_model()
        fmlas = [f for f in Formula.formulas(model, atoms_only=True)]
        logging.info("Minimal quorum: %s", _get_quorum_from_atoms(fmlas, 'A'))
    else:
        logging.info("%s is not in a minimal quorum of %s", v1, v2)
    return res