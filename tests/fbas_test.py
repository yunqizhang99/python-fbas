from python_fbas.fbas import QSet, FBAS, qset_intersection_bound
from test_utils import get_validators_from_test_data_file

q1 = QSet.make(3, [1,2,3,4],[])
o1 = QSet.make(2, [11,12,13],[])
o2 = QSet.make(2, [21,22,23],[])
o3 = QSet.make(2, [31,32,33],[])
q2 = QSet.make(2,['a'],[o1,o2,o3])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})

def test_sat():
    assert q1.sat([1,2,3])
    assert q1.sat([2,3,4])
    assert not q1.sat([1,2])
    assert not q1.sat([4,3])

def test_sat_2():
    assert q2.sat([11,12,32,33])
    assert q2.sat(['a',32,33])
    assert not q2.sat(['a'])
    assert not q2.sat(['a',12,21,33])

def test_slices():
    assert q1.slices() == {frozenset({1,2,3}),frozenset({2,3,4}),frozenset({3,4,1}),frozenset({4,1,2})}
    assert frozenset({11,12,21,22}) in q2.slices()
    assert frozenset({11,13,23,22}) in q2.slices()
    assert frozenset({11,13,33,32}) in q2.slices()
    assert frozenset({'a',33,32}) in q2.slices()

def test_level_1_v_blocking_sets():
    assert q1.blocking_sets() == {frozenset({1,2}),frozenset({2,3}),frozenset({3,4}),frozenset({4,1}),frozenset({1,3}),frozenset({2,4})}
    assert frozenset(({'a',11,12,21,22})) in q2.blocking_sets()
    assert frozenset(({11,13,23,22,31,33})) in q2.blocking_sets()

def test_3():
    assert q1.all_validators() == frozenset({1,2,3,4})
    assert q2.all_validators() == frozenset({'a',11,12,13,21,22,23,31,32,33})
    assert q1.all_qsets() == frozenset({q1})
    assert q2.all_qsets() == frozenset({q2,o1,o2,o3})

def test_blocked():
    assert q1.blocked([1,2])
    assert not q1.blocked([1])
    assert q2.blocked(['a',11,12,32,33])
    assert not q2.blocked(['a',11,32,33])

def test_is_quorum():
    assert fbas1.is_quorum([1,2,4])
    assert not fbas1.is_quorum([2,4])

def test_to_graph():
    assert set(fbas1.to_mixed_graph().nodes) == {1,2,3,4,q1}
    assert set(fbas1.to_mixed_graph().edges) == {(1,q1),(2,q1),(3,q1),(4,q1),(q1,1),(q1,2),(q1,3),(q1,4)}
    assert set(fbas1.to_graph().edges) == {(1,1),(1,2),(1,3),(1,4),(2,1),(2,2),(2,3),(2,4),(3,1),(3,2),(3,3),(3,4),(4,1),(4,2),(4,3),(4,4)}

def test_closure():
    assert fbas1.closure([1,2]) == {1,2,3,4}
    assert fbas1.closure([2]) == {2}

def test_depth():
    assert q1.depth() == 1
    assert q2.depth() == 2

def test_stellarbeat():
    fbas = FBAS.from_json(get_validators_from_test_data_file('validators.json'))
    fbas.to_mixed_graph()
    fbas.to_graph()
    FBAS.from_json(get_validators_from_test_data_file('validators_broken_1.json'))

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
    assert qset_intersection_bound(qset_1, qset_2) == 2
    assert qset_intersection_bound(qset_1, qset_3) == 1
    assert qset_intersection_bound(qset_1, qset_4) == 1
    fbas = FBAS.from_json(get_validators_from_test_data_file('validators.json'))
    assert fbas.min_scc_intersection_bound() == 3

def test_is_org_structured():
    assert fbas1.is_org_structured()
    q = QSet.make(2,[],[o1,o2,o3])
    fbas2 = FBAS({11 : q, 12 : q, 13 : q, 21 : q, 22 : q, 23 : q, 31 : q, 32 : q, 33 : q})
    assert fbas2.is_org_structured()
    q3 = QSet.make(2,['x'],[o1,o2,o3])
    fbas3 = FBAS({11: q3, 12: q3, 13: q3, 21: q3, 22: q3, 23: q3, 31: q3, 32: q3, 33: q3, 'x': QSet.make(1,[11],[])})
    assert not fbas3.is_org_structured()

def test_weights():
    g = fbas1.to_weighed_graph()
    assert len(g.edges) == 16
    # iterate over the graph and check that all edges have a weight of 1/4:
    for (u,v) in g.edges:
        assert g[u][v]['weight'] == 1/4
    q = QSet.make(2,[],[o1,o2,o3])
    fbas2 = FBAS({11 : q, 12 : q, 13 : q, 21 : q, 22 : q, 23 : q, 31 : q, 32 : q, 33 : q})
    g2 = fbas2.to_weighed_graph()
    for (u,v) in g2.edges:
        assert g2[u][v]['weight'] == 1/9

def test_collapse_qsets():
    collapsed_fbas = fbas1.collapse_qsets()
    qs = QSet.make(1, [0], [])
    assert collapsed_fbas == FBAS({1: qs, 2: qs, 3: qs, 4: qs, 0: qs})
    stellar = FBAS.from_json(get_validators_from_test_data_file('validators.json'))
    collapsed_stellar = stellar.collapse_qsets(new_name=stellar.org_of_qset)
    assert 'www.stellar.org' in collapsed_stellar.qset_map.keys()
