import logging
from pysat.formula import *
from pysat.examples.lsu import LSU
from typing import Optional

from .fbas import QSet, FBAS
from .utils import to_cnf, clause_of_pseudo_atom

def _constellation_groups(fbas : FBAS) -> list[int]:
    return list(range(len(fbas.meta_field_values('homeDomain'))))

def _constellation_constraints(fbas : FBAS):
    # first make sure that all validators have a homeDomain:
    if not fbas.all_have_meta_field('homeDomain'):
        raise ValueError("Some validators do not have a homeDomain")
    # also check that each validator of an org has the same QSet, and that this QSet is of the form k-out-of-n orgs:
    orgs = fbas.meta_field_values('homeDomain')
    for org in orgs:
        qsets = {v : fbas.qset_map[v] for v in fbas.validators_with_meta_field_value('homeDomain', org)}
        if len(set(qsets.values())) > 1:
            # logging.error(f"Validators of org {org} have {len(set(qsets.values()))} different QSets:\n{chr(10).join(fbas.validator_name(v) + fbas.format_qset(q) for v,q in qsets.items())}")
            logging.error(f"Validators of org {org} have {len(set(qsets.values()))} different QSets:\n{chr(10).join(str(q) for q in set(qsets.values()))}")
            raise ValueError(f"Validators of org {org} have {len(set(qsets.values()))} different QSets")
        if list(qsets.values()).pop().validators:
            raise ValueError(f"QSet of org {org} is not of the right form")

    constraints : list[Formula] = []
    # create groups 1 to n where n is the number of distinct homeDomains:
    groups = _constellation_groups(fbas)
    # every org is in exactly one group:
    def in_group(g, o):
        return And(Atom(('in-group', g, o)), Neg(Or(*[Atom(('in-group', g2, o)) for g2 in set(groups) - {g}])))
    constraints += [Or(*[in_group(g, o) for g in groups]) for o in orgs]
    # each group is a clique:
    constraints += [
        Implies(And(in_group(g, o1), in_group(g, o2)), Atom(('edge', o1, o2)))
            for g in groups for o1 in orgs for o2 in orgs
            if o1 != o2
    ]
    # each org has an edge to each other non-empty group: 
    constraints += [
        Implies(
            And(in_group(g1, o1), Or(*[in_group(g2, o2) for o2 in orgs])),
            Or(*[And(in_group(g2, o2), Atom(('edge', o1, o2))) for o2 in orgs])
        )
        for g1 in groups for g2 in groups if g1 != g2 for o1 in orgs
    ]
    # each org has enough edges to satisfy its failure assumptions:
    def connected_to_all_in(o, b):
        return And(*[Atom(('edge', o, neighbor)) for neighbor in b if neighbor != o])
    def qset_of_org(o):
        return fbas.qset_map[fbas.validators_with_meta_field_value('homeDomain', o).pop()]
    constraints += [
        Or(*[connected_to_all_in(o, b) for b in qset_of_org(o).blocking_sets()]) for o in orgs
    ]
    # now minimize the number of edges:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    for o1 in orgs:
        for o2 in orgs - {o1}:
            f = Neg(Atom(('edge', o1, o2)))
            wcnf.append(clause_of_pseudo_atom(f), weight=1)
    return wcnf

def constellation_graph(fbas, solver_class=LSU) -> Optional[dict]:
    wcnf = _constellation_constraints(fbas)
    maxSAT_solver = solver_class(wcnf)
    got_model = (
        maxSAT_solver.compute() if 'compute' in solver_class.__dict__ else maxSAT_solver.solve()
    )
    if got_model:
        print(f'Found constellation graph of cost {maxSAT_solver.cost}')
        model = maxSAT_solver.model
        fmlas = Formula.formulas(model, atoms_only=True)
        orgs = fbas.meta_field_values('homeDomain')
        groups = {
            g : {o for o in orgs
                    if Atom(('in-group', g, o)) in fmlas} for g in _constellation_groups(fbas)}
        print ("groups: ", groups)
        return groups
    else:
        raise ValueError("No constellation graph found")

def _optimal_overlay_constraints(fbas : FBAS):
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
    constraints += [Or(*[And(*[edge(v, neighbor) for neighbor in blocking]) for blocking in fbas.qset_map[v].level_1_v_blocking_sets()])
        for v in fbas.validators()]
    # minimize the number of edges:
    wcnf = WCNF()
    wcnf.extend(to_cnf(constraints))
    for v1 in fbas.validators():
        for v2 in fbas.validators() - {v1}:
            f = Neg(edge(v1,v2))
            wcnf.append(clause_of_pseudo_atom(f), weight=1)
    return wcnf

def optimal_overlay(fbas, solver_class=LSU):  # LSU seems to perform the best
    wcnf = _optimal_overlay_constraints(fbas)
    print("computed constraints")
    maxSAT_solver = solver_class(wcnf)
    got_model = (
        maxSAT_solver.compute() if 'compute' in solver_class.__dict__ else maxSAT_solver.solve()
    )
    if got_model:
        print(f'Found minimal overlay of cost {maxSAT_solver.cost}')
        model = maxSAT_solver.model
        fmlas = Formula.formulas(model, atoms_only=True)
        edges = [a.object for a in fmlas if isinstance(a, Atom)]
        # remove symmetric edges:
        edges = [edge for edge in edges if edge[0] < edge[1]]
        print("edges:", edges)
        return frozenset(edges)
    else:
        return frozenset()