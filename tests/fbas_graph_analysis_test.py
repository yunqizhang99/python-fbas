import logging
from test_utils import get_test_data_list, get_validators_from_test_fbas
from python_fbas.fbas_graph import FBASGraph
from python_fbas import config
from python_fbas.fbas_graph_analysis import find_disjoint_quorums_using_pysat_fmla, find_disjoint_quorums, find_minimal_splitting_set, find_minimal_blocking_set, find_disjoint_quorums_, find_minimal_splitting_set_
from python_fbas.z3_fbas_graph_analysis import find_disjoint_quorums as z3_find_disjoint_quorums

def test_qi_pysat_fmla():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums_using_pysat_fmla(fbas)
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums_using_pysat_fmla(fbas)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert find_disjoint_quorums_using_pysat_fmla(fbas2)

def test_qi_z3():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not z3_find_disjoint_quorums(fbas)
    fbas3 = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not z3_find_disjoint_quorums(fbas3)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert z3_find_disjoint_quorums(fbas2)

def test_qi_cnf():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums(fbas)
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums(fbas)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert find_disjoint_quorums(fbas2)

def test_qi_():
    config.card_encoding = 'totalizer'
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums_(fbas)
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums_(fbas)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert find_disjoint_quorums_(fbas2)

def test_qi_2_():
    config.card_encoding = 'naive'
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums_(fbas)
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums_(fbas)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert find_disjoint_quorums_(fbas2)

def test_qi_all():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            assert (not find_disjoint_quorums_using_pysat_fmla(fbas_graph)) == (not find_disjoint_quorums(fbas_graph))
            assert (not find_disjoint_quorums(fbas_graph)) == (not z3_find_disjoint_quorums(fbas_graph))
            config.card_encoding = 'totalizer'
            assert (not find_disjoint_quorums_(fbas_graph)) == (not z3_find_disjoint_quorums(fbas_graph))
            config.card_encoding = 'naive'
            assert (not find_disjoint_quorums_(fbas_graph)) == (not z3_find_disjoint_quorums(fbas_graph))
    
def test_min_splitting_set_1():
    qset1 = {'threshold':3, 'validators':['PK1','PK2','PK3','PK4'],  'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1','PK2','PK3','PK4']:
        fbas1.update_validator(v, qset1)
    assert len(find_minimal_splitting_set(fbas1)[0]) == 2
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_minimal_splitting_set(fbas2)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert find_minimal_splitting_set(fbas2)[0] == ['PK2']
        
def test_min_splitting_set_2():
    qset1 = {'threshold':3, 'validators':['PK1','PK2','PK3','PK4'],  'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1','PK2','PK3','PK4']:
        fbas1.update_validator(v, qset1)
    assert len(find_minimal_splitting_set_(fbas1)[0]) == 2
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_minimal_splitting_set_(fbas2)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert find_minimal_splitting_set_(fbas2)[0] == ['PK2']


def test_min_blocking_set_1():
    qset1 = {'threshold':3, 'validators':['PK1','PK2','PK3','PK4'],  'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1','PK2','PK3','PK4']:
        fbas1.update_validator(v, qset1)
    config.card_encoding = 'totalizer'
    config.max_sat_algo = 'RC2'
    b = find_minimal_blocking_set(fbas1)
    assert len(b) == 2


def test_min_blocking_set_2():
    data = get_test_data_list()
    for f,d in data.items():
        if f == 'top_tier.json':
            logging.info("loading graph of %s", f)
            fbas_graph = FBASGraph.from_json(d)
            config.card_encoding = 'totalizer'
            find_minimal_blocking_set(fbas_graph)