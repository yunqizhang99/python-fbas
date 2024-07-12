from timeit import *
from intersection_checker import min_splitting_set, check_intersection
from stellarbeat import get_validators_from_file
from fbas import FBAS
from test_utils import get_test_data_file_path

def benchmark():
    file = 'almost_symmetric_network_16_orgs_delete_prob_factor_1.json'
    fbas = FBAS.from_stellarbeat_json(get_validators_from_file(get_test_data_file_path(file)))
    # if not check_intersection(fbas):
        # print("Intersection failed")
    t = timeit(lambda: min_splitting_set(fbas), number=1)
    print(t)

if __name__ == '__main__':
    benchmark()