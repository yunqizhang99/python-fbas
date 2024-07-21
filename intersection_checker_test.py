from sat_based_fbas_analysis import *
from fbas import QSet, FBAS
from test_utils import get_test_data_file_path, get_validators_from_file

q1 = QSet.make(3, [1,2,3,4],[])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})
q2 = QSet.make(2, [1,2,3,4],[])
fbas2 = FBAS({1 : q2, 2 : q2, 3 : q2, 4 : q2})
fbas = FBAS.from_stellarbeat_json(get_validators_from_file(get_test_data_file_path('validators.json')))

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

def test_optimal_overlay():
    assert len(optimal_overlay(fbas1)) == 5
    assert len(optimal_overlay(fbas2)) == 6
