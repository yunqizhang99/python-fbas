from fbas import *
import unittest

q1 = QSet.make(3, [1,2,3,4],[])
o1 = QSet.make(2, [11,12,13],[])
o2 = QSet.make(2, [21,22,23],[])
o3 = QSet.make(2, [31,32,33],[])
q2 = QSet.make(2,['a'],[o1,o2,o3])

class QSetTest(unittest.TestCase):

    def test_1(self):
        self.assertTrue(q1.sat([1,2,3]))
        self.assertTrue(q1.sat([2,3,4]))
        self.assertFalse(q1.sat([1,2]))
        self.assertFalse(q1.sat([4,3]))

    def test_2(self):
        self.assertTrue(q2.sat([11,12,32,33]))
        self.assertTrue(q2.sat(['a',32,33]))
        self.assertFalse(q2.sat(['a']))
        self.assertFalse(q2.sat(['a',12,21,33]))

    def test_3(self):
        self.assertTrue(q1.all_validators() == frozenset({1,2,3,4}))
        self.assertTrue(q2.all_validators() == frozenset({'a',11,12,13,21,22,23,31,32,33}))
        self.assertTrue(q1.all_qsets() == frozenset({q1}))
        self.assertTrue(q2.all_qsets() == frozenset({q2,o1,o2,o3}))

    def test_4(self):
        self.assertTrue(q1.blocked([1,2]))
        self.assertFalse(q1.blocked([1]))
        self.assertTrue(q2.blocked(['a',11,12,32,33]))
        self.assertFalse(q2.blocked(['a',11,32,33]))

fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})

class FBASTest(unittest.TestCase):

    def test_1(self):
        self.assertTrue(fbas1.is_quorum([1,2,4]))
        self.assertFalse(fbas1.is_quorum([2,4]))

    def test_2(self):
        self.assertTrue(set(fbas1.to_graph().nodes) == {1,2,3,4,q1})
        self.assertEqual(set(fbas1.to_graph().edges), {(1,q1),(2,q1),(3,q1),(4,q1),(q1,1),(q1,2),(q1,3),(q1,4)})

    def test_3(self):
        self.assertTrue(fbas1.closure([1,2]) == {1,2,3,4})
        self.assertTrue(fbas1.closure([2]) == {2})