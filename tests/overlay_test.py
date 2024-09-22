from python_fbas.overlay import constellation_graph, optimal_overlay
from python_fbas.fbas import FBAS, QSet
from test_utils import get_validators_from_test_data_file

q1 = QSet.make(3, [1,2,3,4],[])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})
q2 = QSet.make(2, [1,2,3,4],[])
fbas2 = FBAS({1 : q2, 2 : q2, 3 : q2, 4 : q2})

def test_optimal_overlay():
    assert len(optimal_overlay(fbas1)) == 5
    assert len(optimal_overlay(fbas2)) == 6

# def test_constellation_1():
    # fbas = FBAS.from_json(get_validators_from_test_data_file('homedomain_test_1.json'))
    # g = constellation_graph(fbas)