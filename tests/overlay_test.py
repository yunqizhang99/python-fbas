from python_fbas.overlay import constellation_graph, optimal_overlay
from python_fbas.fbas import FBAS, QSet
from python_fbas.fbas_generator import gen_symmetric_fbas
from python_fbas.sat_based_fbas_analysis import is_in_min_quorum_of
from test_utils import get_validators_from_test_data_file

q1 = QSet.make(3, [1,2,3,4],[])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})
q2 = QSet.make(2, [1,2,3,4],[])
fbas2 = FBAS({1 : q2, 2 : q2, 3 : q2, 4 : q2})

def test_optimal_overlay():
    assert len(optimal_overlay(fbas1)) == 5
    assert len(optimal_overlay(fbas2)) == 6

def test_constellation_1():
    fbas = gen_symmetric_fbas(3)
    g = constellation_graph(fbas)
    print(g)

def test_is_in_min_quorum_of_1():
    fbas = gen_symmetric_fbas(3)
    for v1 in fbas.validators():
        for v2 in fbas.validators():
            assert is_in_min_quorum_of(fbas, v1, v2)

def test_is_in_min_quorum_of_2():
    q1 = QSet.make(1, [2],[])
    q2 = QSet.make(1, [2],[])
    fbas = FBAS({1 : q1, 2 : q2})
    # TODO: debug!
    assert is_in_min_quorum_of(fbas, 1, 2)
    assert not is_in_min_quorum_of(fbas, 2, 1)
