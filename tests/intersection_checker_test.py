from python_fbas.sat_based_fbas_analysis import check_intersection, min_splitting_set, min_blocking_set, min_blocking_set_mus, is_in_min_quorum_of
from python_fbas.fbas import QSet, FBAS
from python_fbas.fbas_generator import gen_symmetric_fbas
from test_utils import get_validators_from_test_data_file

q1 = QSet.make(3, [1,2,3,4],[])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})
q2 = QSet.make(2, [1,2,3,4],[])
fbas2 = FBAS({1 : q2, 2 : q2, 3 : q2, 4 : q2})
fbas = FBAS.from_json(get_validators_from_test_data_file('validators.json')).sanitize()

def test_1():
    assert check_intersection(fbas1)
    assert not check_intersection(fbas2)

def test_2():
    assert check_intersection(fbas)

# TODO: check that the disjoint quorums found are actually quorums
def test_min_splitting_set():
    assert len(min_splitting_set(fbas)) == 3
    assert len(min_splitting_set(fbas1)) == 2
    assert len(min_splitting_set(fbas2)) == 0

def test_min_blocking_set_():
    # too slow:
    # assert len(min_blocking_set(fbas)) == 6
    assert len(min_blocking_set(fbas1)) == 2
    assert len(min_blocking_set(fbas2)) == 3

def test_min_blocking_set_mus():
    assert len(min_blocking_set_mus(fbas)) == 6
    assert len(min_blocking_set_mus(fbas1)) == 2
    assert len(min_blocking_set_mus(fbas2)) == 3

def test_group_by_1():
    fbas = FBAS.from_json(get_validators_from_test_data_file('homedomain_test_1.json'))
    assert min_splitting_set(fbas, group_by='homeDomain') == ["domain-2"]

def test_is_in_min_quorum_of_1():
    fbas = gen_symmetric_fbas(3)
    for v1 in fbas.validators():
        for v2 in fbas.validators():
            assert is_in_min_quorum_of(fbas, v1, v2)

def test_is_in_min_quorum_of_2():
    q1 = QSet.make(1, [2],[])
    q2 = QSet.make(1, [2],[])
    fbas = FBAS({1 : q1, 2 : q2})
    assert is_in_min_quorum_of(fbas, 1, 2)
    assert not is_in_min_quorum_of(fbas, 2, 1)