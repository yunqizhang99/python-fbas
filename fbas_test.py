import pytest
from fbas import *
from stellarbeat import get_validators_from_file
from test_utils import get_test_data_file_path

q1 = QSet.make(3, [1,2,3,4],[])
o1 = QSet.make(2, [11,12,13],[])
o2 = QSet.make(2, [21,22,23],[])
o3 = QSet.make(2, [31,32,33],[])
q2 = QSet.make(2,['a'],[o1,o2,o3])

def test_1():
    assert q1.sat([1,2,3])
    assert q1.sat([2,3,4])
    assert not q1.sat([1,2])
    assert not q1.sat([4,3])

def test_2():
    assert q2.sat([11,12,32,33])
    assert q2.sat(['a',32,33])
    assert not q2.sat(['a'])
    assert not q2.sat(['a',12,21,33])

def test_3():
    assert q1.all_validators() == frozenset({1,2,3,4})
    assert q2.all_validators() == frozenset({'a',11,12,13,21,22,23,31,32,33})
    assert q1.all_qsets() == frozenset({q1})
    assert q2.all_qsets() == frozenset({q2,o1,o2,o3})

def test_4():
    assert q1.blocked([1,2])
    assert not q1.blocked([1])
    assert q2.blocked(['a',11,12,32,33])
    assert not q2.blocked(['a',11,32,33])

fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})

def test_5():
    assert fbas1.is_quorum([1,2,4])
    assert not fbas1.is_quorum([2,4])

def test_6():
    assert set(fbas1.to_mixed_graph().nodes) == {1,2,3,4,q1}
    assert set(fbas1.to_mixed_graph().edges) == {(1,q1),(2,q1),(3,q1),(4,q1),(q1,1),(q1,2),(q1,3),(q1,4)}
    assert set(fbas1.to_graph().edges) == {(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)}

def test_7():
    assert fbas1.closure([1,2]) == {1,2,3,4}
    assert fbas1.closure([2]) == {2}

def test_depth():
    assert q1.depth() == 1
    assert q2.depth() == 2

def test_stellarbeat():
    fbas = FBAS.from_stellarbeat_json(get_validators_from_file(get_test_data_file_path('validators.json')))
    fbas.to_mixed_graph()
    fbas.to_graph()
    with pytest.raises(Exception):
        FBAS.from_stellarbeat_json(get_validators_from_file(get_test_data_file_path('validators_broken_1.json')))

def test_min_direct_intersection():
    org_a = QSet.make(2, ['a1','a2','a3'],[])
    org_b = QSet.make(2, ['b1','b2','b3'],[])
    org_c = QSet.make(2, ['c1','c2','c3'],[])
    org_d = QSet.make(2, ['d1','d2','d3'],[])
    org_e = QSet.make(2, ['e1','e2','e3'],[])
    org_f = QSet.make(2, ['f1','f2','f3'],[])
    org_g = QSet.make(2, ['g1','g2','g3'],[])
    qset_1 = QSet.make(4, [], [org_a, org_b, org_c, org_d, org_e])
    qset_2 = QSet.make(4, [], [org_b, org_c, org_d, org_e, org_f])
    qset_3 = QSet.make(4, [], [org_c, org_d, org_e, org_f, org_g])
    qset_4 = QSet.make(4, ['x'], [org_c, org_d, org_e, org_f])
    assert min_direct_intersection(qset_1, qset_2) == 2
    assert min_direct_intersection(qset_1, qset_3) == 1
    assert min_direct_intersection(qset_1, qset_4) == 1
    fbas = FBAS.from_stellarbeat_json(get_validators_from_file(get_test_data_file_path('validators.json')))
    assert fbas.min_scc_direct_intersection() == 3
