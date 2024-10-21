import logging
import sys
from pysat.formula import *
from pysat.examples.lsu import LSU
from pysat.card import *
from typing import Optional
from python_fbas.fbas_generator import gen_symmetric_fbas, gen_asymmetric_fbas

from python_fbas.fbas import FBAS
from python_fbas.utils import to_cnf, clause_of_pseudo_atom

def _constellation_groups(fbas : FBAS) -> list[int]:
    upper = int(len(fbas.meta_field_values('homeDomain'))/3)+2
    logging.info(f"Using max {upper} groups")
    return list(range(upper))

def _constellation_constraints(fbas : FBAS, card_encoding=EncType.pairwise):
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

    wcnf = WCNF()
    constraints : list[Formula] = []
    # create groups 1 to n where n is the number of distinct homeDomains:
    groups = _constellation_groups(fbas)
    def in_group(g, o):
        # return And(Atom(('in-group', g, o)), Neg(Or(*[Atom(('in-group', g2, o)) for g2 in set(groups) - {g}])))
        return Atom(('in-group', g, o))
    # each group is a clique:
    constraints += [
        Implies(And(in_group(g, o1), in_group(g, o2)), Atom(('edge', o1, o2)))
            for g in groups for o1 in orgs for o2 in orgs
            if o1 != o2
    ]

    # each org has at least one edge to each other non-empty group: 
    constraints += [
        Implies(
            And(in_group(g1, o1), Or(*[in_group(g2, o2) for o2 in orgs])),
            Or(*[And(in_group(g2, o2), Atom(('edge', o1, o2))) for o2 in orgs if o2 != o1])
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
    wcnf.extend(to_cnf(constraints))

    # each org is in exactly one group:
    wcnf.extend(to_cnf([Or(*[in_group(g, o) for g in groups]) for o in orgs]))
    vpool = Formula.export_vpool()
    for o in orgs:
        atoms = [in_group(g, o) for g in groups]
        for a in atoms:
            a.clausify()
        lits = [a.clauses[0][0] for a in atoms]
        wcnf.extend(CardEnc.atmost(lits=lits, vpool=vpool, encoding=card_encoding).clauses) 
        # NOTE: specifying the vpool is crucial, otherwise the encoding will not make any sense

    for o1 in orgs:
        for o2 in orgs - {o1}:
            f = Neg(Atom(('edge', o1, o2)))
            wcnf.append(clause_of_pseudo_atom(f), weight=1)
    return wcnf

def constellation_graph(fbas, solver_class=LSU, card_encoding=EncType.pairwise) -> Optional[dict]:
    wcnf = _constellation_constraints(fbas, card_encoding)
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
    
if __name__ == '__main__':
    # setup logging:
    logging.basicConfig(level=logging.INFO)
    the_fbas = gen_asymmetric_fbas(8)
    # pretty-print the json:
    # fbas_json = fbas.to_json()
    # pretty = json.dumps(fbas_json, indent=4)
    # print(pretty)
    print(constellation_graph(the_fbas))
    # print(f"Found optimal overlay with {len(optimal_overlay(fbas))/2} edges")
    sys.exit(0)