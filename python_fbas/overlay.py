import logging
from pysat.formula import *
from pysat.examples.lsu import LSU
from pysat.card import *
from typing import Optional

from .fbas import QSet, FBAS
from .utils import to_cnf, clause_of_pseudo_atom

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