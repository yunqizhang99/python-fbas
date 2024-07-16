from timeit import *
from sat_based_fbas_analysis import min_splitting_set, check_intersection, min_blocking_set
from stellarbeat import get_validators_from_file
from fbas import FBAS
from test_utils import get_test_data_file_path
from pysat.examples.fm import FM
from pysat.examples.lsu import LSU
from pysat.examples.rc2 import RC2

solvers = LSU, RC2  # FM is really slow

def benchmark():
    file = 'almost_symmetric_network_16_orgs.json'
    # file = 'almost_symmetric_network_10_orgs.json'
    fbas = FBAS.from_stellarbeat_json(get_validators_from_file(get_test_data_file_path(file)))
    # if not check_intersection(fbas):
        # print("Intersection failed")
    for s in solvers:
        print(s)
        t = timeit(lambda: min_splitting_set(fbas, solver_class=s), number=1)
        print(t)

if __name__ == '__main__':
    benchmark()