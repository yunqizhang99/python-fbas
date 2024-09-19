from timeit import *
from python_fbas.sat_based_fbas_analysis import min_splitting_set, check_intersection, min_blocking_set
from tests.test_utils import get_validators_from_test_data_file
from python_fbas.fbas import FBAS
from pysat.examples.fm import FM
from pysat.examples.lsu import LSU
from pysat.examples.rc2 import RC2

solvers = LSU , # RC2  # FM is really slow

def benchmark():
    # file = 'almost_symmetric_network_16_orgs.json'
    file = 'almost_symmetric_network_16_orgs_delete_prob_factor_1.json'
    fbas = FBAS.from_json(get_validators_from_test_data_file(file))
    # if not check_intersection(fbas):
        # print("Intersection failed")
    for s in solvers:
        print(s)
        # t = timeit(lambda: min_splitting_set(fbas, solver_class=s), number=1)
        t = timeit(lambda: min_blocking_set(fbas), number=1)
        print(t)

if __name__ == '__main__':
    benchmark()