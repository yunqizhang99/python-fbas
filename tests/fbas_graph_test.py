import logging
import pytest
import random
from test_utils import get_test_data_list, get_validators_from_test_fbas
from python_fbas.fbas_graph import FBASGraph

def test_collapse():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fg = FBASGraph.from_json(d)
        fg.check_integrity()
        logging.info("graph of %s before flattening:\n %s", f, fg.stats())
        fg.flatten_diamonds()
        logging.info("graph of %s after flattening:\n %s", f, fg.stats())

def test_is_quorum():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert fbas.is_quorum({'PK11','PK12','PK13'})
    assert fbas.is_quorum({'PK11','PK12'})
    assert not fbas.is_quorum({'PK11'})
    assert not fbas.is_quorum({'PK11','PK12','PK13','PK21'})
    assert not fbas.is_quorum({'PK13','PK21'})
    assert fbas.is_quorum({'PK11','PK12','PK13','PK21','PK22','PK23'})
    with pytest.raises(AssertionError):
        fbas.is_quorum({'PK11','PK12','PK13','NON_EXISTENT'})
    assert not fbas.is_quorum({'PK11','PK12','PK13','PKX'})
    assert fbas.is_quorum({'PK11','PK12','PKX','PK22','PK23'})
    assert not fbas.is_quorum({'PK11','PK12','PKX','PK22'})

def test_is_quorum_2():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            for _ in range(100):
                # pick a random subset of validators for which we have a qset:
                vs = [v for v in fbas_graph.validators if fbas_graph.threshold(v) > 0]
                n = random.randint(1, len(vs))
                validators = random.sample(vs, n)
                fbas_graph.is_quorum(validators)

def test_is_sat():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert fbas.is_sat('PK3', {'PK3'})
    assert fbas.is_sat('PK2', {'PK3'})
    assert not fbas.is_sat('PK1', {'PK3'})

def test_find_disjoint_quorums():
    fbas1 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    q1, q2 = fbas1.find_disjoint_quorums()
    logging.info("disjoint quorums: %s, %s", q1, q2)
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('circular_1.json'))
    assert not fbas2.find_disjoint_quorums()
    fbas3 = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert not fbas3.find_disjoint_quorums()

def test_closure():
    data = get_test_data_list()
    for f,d in data.items():
        logging.info("loading graph of %s", f)
        fbas_graph = FBASGraph.from_json(d)
        if fbas_graph.validators:
            for _ in range(100):
                # pick a random subset of validators:
                n = random.randint(1, len(fbas_graph.validators))
                validators = random.sample(list(fbas_graph.validators), n)
                assert fbas_graph.closure(validators)

def test_closure_2():
    fbas = FBASGraph.from_json(get_validators_from_test_fbas('circular_2.json'))
    assert fbas.closure({'PK3'}) == {'PK1', 'PK2', 'PK3'}
    assert fbas.closure({'PK2'}) == {'PK1', 'PK2'}
    assert fbas.closure({'PK1'}) == {'PK1'}
    assert fbas.closure({'PK2','PK3'}) == {'PK1', 'PK2', 'PK3'}
    fbas2 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert fbas2.closure({'PK11','PK12'}) == {'PK11','PK12','PK13','PKX'}
    assert fbas2.closure({'PKX'}) == {'PKX'}
    assert fbas2.closure({'PK11','PK22'}) == {'PK11','PK22'}