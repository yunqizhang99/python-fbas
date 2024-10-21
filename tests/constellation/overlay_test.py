from python_fbas.constellation.overlay import optimal_overlay
from python_fbas.fbas import FBAS, QSet
from python_fbas.fbas_generator import gen_symmetric_fbas

q1 = QSet.make(3, [1,2,3,4],[])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})
q2 = QSet.make(2, [1,2,3,4],[])
fbas2 = FBAS({1 : q2, 2 : q2, 3 : q2, 4 : q2})

def test_optimal_overlay():
    assert len(optimal_overlay(fbas1))/2 == 5
    assert len(optimal_overlay(fbas2))/2 == 6