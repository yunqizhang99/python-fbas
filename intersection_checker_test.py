import os
import unittest
from intersection_checker import check_intersection
from fbas import QSet, FBAS
from stellarbeat import get_validators_from_file

q1 = QSet.make(3, [1,2,3,4],[])
fbas1 = FBAS({1 : q1, 2 : q1, 3 : q1, 4 : q1})
q2 = QSet.make(2, [1,2,3,4],[])
fbas2 = FBAS({1 : q2, 2 : q2, 3 : q2, 4 : q2})

def test_file(name):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'test_data', name)

class IntersectionCheckerTest(unittest.TestCase):
    def test_1(self):
        self.assertTrue(check_intersection(fbas1))
        self.assertFalse(check_intersection(fbas2))

    def test_2(self):
        fbas = FBAS.from_stellarbeat_json(get_validators_from_file(test_file('validators.json')))
        self.assertTrue(check_intersection(fbas))

    
