from timeit import *
from python_fbas.sat_based_fbas_analysis import optimal_overlay

from test_utils import get_validators_from_test_data_file
from python_fbas.fbas import FBAS

def benchmark():
    # file = 'almost_symmetric_network_12_orgs.json'
    file = 'almost_symmetric_network_5_orgs_delete_prob_factor_1.json'
    fbas = FBAS.from_json(get_validators_from_test_data_file(file))
    t = timeit(lambda: optimal_overlay(fbas), number=1)
    print(t)

if __name__ == '__main__':
    benchmark()