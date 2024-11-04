from python_fbas.sat_based_fbas_analysis import check_intersection, min_splitting_set, min_blocking_set, min_blocking_set_mus
from python_fbas.fbas import QSet, FBAS
from test_utils import get_validators_from_test_fbas

q1 = QSet.make(3, [1,2,3,4],[])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})
q2 = QSet.make(2, [1,2,3,4],[])
fbas2 = FBAS({1 : q2, 2 : q2, 3 : q2, 4 : q2})
fbas = FBAS.from_json(get_validators_from_test_fbas('validators.json')).sanitize()

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
    fbas = FBAS.from_json(get_validators_from_test_fbas('homedomain_test_1.json'))
    assert min_splitting_set(fbas, group_by='homeDomain') == ["domain-2"]

def test_empty_qset():
    circular_fbas = FBAS.from_json(get_validators_from_test_fbas('circular_2.json'))
    # TODO: fails when using collapse_qsets
    assert check_intersection(circular_fbas)