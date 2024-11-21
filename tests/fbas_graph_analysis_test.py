import logging
from test_utils import get_test_data_list, get_validators_from_test_fbas
from python_fbas.fbas_graph import FBASGraph
from python_fbas.fbas_graph_analysis import find_disjoint_quorums_using_pysat_fmla, find_disjoint_quorums, find_minimal_splitting_set
from python_fbas.z3_fbas_graph_analysis import find_disjoint_quorums as z3_find_disjoint_quorums

def test_quorum_intersection():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums_using_pysat_fmla(fbas)
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums_using_pysat_fmla(fbas)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert find_disjoint_quorums_using_pysat_fmla(fbas2)

def test_quorum_intersection_z3():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not z3_find_disjoint_quorums(fbas)
    fbas3 = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not z3_find_disjoint_quorums(fbas3)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert z3_find_disjoint_quorums(fbas2)

def test_quorum_intersection_cnf():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums(fbas)
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums(fbas)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert find_disjoint_quorums(fbas2)

def test_quorum_intersection_2():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            find_disjoint_quorums_using_pysat_fmla(fbas_graph)

def test_quorum_intersection_z3_2():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            z3_find_disjoint_quorums(fbas_graph)

def test_quorum_intersection_cnf_2():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            find_disjoint_quorums(fbas_graph)

def test_compare():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            assert (not find_disjoint_quorums_using_pysat_fmla(fbas_graph)) == (not find_disjoint_quorums(fbas_graph))

def test_compare_z3():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            assert (not find_disjoint_quorums_using_pysat_fmla(fbas_graph)) == (not z3_find_disjoint_quorums(fbas_graph))

def test_quorum_intersection_3():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            find_disjoint_quorums_using_pysat_fmla(fbas_graph, flatten=True)

def test_quorum_intersection_4():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_disjoint_quorums_using_pysat_fmla(fbas, flatten=True)
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not find_disjoint_quorums_using_pysat_fmla(fbas, flatten=True)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted_2.json'))
    q1, q2 = fbas2.find_disjoint_quorums()
    logging.info("disjoint quorums: %s, %s", q1, q2)
    assert find_disjoint_quorums_using_pysat_fmla(fbas2, flatten=True)
    
def test_min_splitting_set_1():
    qset1 = {'threshold':3, 'validators':['PK1','PK2','PK3','PK4'],  'innerQuorumSets': []}
    fbas1 = FBASGraph()
    for v in ['PK1','PK2','PK3','PK4']:
        fbas1.update_validator(v, qset1)
    assert len(find_minimal_splitting_set(fbas1)) == 2
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not find_minimal_splitting_set(fbas2)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert find_minimal_splitting_set(fbas2) == ['PK2']