import random
from test_utils import get_validators_from_test_fbas
from python_fbas.deprecated.overlay import optimal_overlay
from python_fbas.deprecated.fbas import FBAS, QSet
from python_fbas.deprecated.fbas_generator import gen_symmetric_fbas
from python_fbas.deprecated.sat_based_fbas_analysis import is_in_min_quorum_of

q1 = QSet.make(3, [1,2,3,4],[])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})
q2 = QSet.make(2, [1,2,3,4],[])
fbas2 = FBAS({1 : q2, 2 : q2, 3 : q2, 4 : q2})

def test_optimal_overlay():
    assert len(optimal_overlay(fbas1)) == 5
    assert len(optimal_overlay(fbas2)) == 6

def test_is_in_min_quorum_of_1():
    fbas = gen_symmetric_fbas(3)
    # pick randomly a validator in fbas.validators():
    v1 = random.choice(list(fbas.validators()))
    v2 = random.choice(list(fbas.validators()))
    assert is_in_min_quorum_of(fbas, v1, v2)

def test_is_in_min_quorum_of_3():
    fbas = FBAS.from_json(get_validators_from_test_fbas('top_tier.json'))
    v1 = random.choice(list(fbas.validators()))
    v2 = random.choice(list(fbas.validators()))
    assert is_in_min_quorum_of(fbas, v1, v2)


def test_is_in_min_quorum_of_2():
    q = QSet.make(1, [2],[])
    fbas = FBAS({1 : q, 2 : q})
    assert is_in_min_quorum_of(fbas, 2, 2)
    assert is_in_min_quorum_of(fbas, 2, 1)
    assert not is_in_min_quorum_of(fbas, 1, 2)

def test_is_in_min_quorum_of_4():
    fbas = FBAS.from_json(get_validators_from_test_fbas('validators.json'))
    astro1 = "GDMAU3NHV4H7NZF5PY6O6SULIUKIIHPRYOKM7HMREK4BW65VHMDKNM6M"
    sdf2 = "GCM6QMP3DLRPTAZW2UZPCPX2LF3SXWXKPMP3GKFZBDSF3QZGV2G5QSTK"
    assert is_in_min_quorum_of(fbas, sdf2, astro1)
    assert not is_in_min_quorum_of(fbas, astro1, sdf2)
