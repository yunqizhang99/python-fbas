from timeit import *
from sat_based_fbas_analysis import optimal_overlay

from stellarbeat import get_validators_from_file
from fbas import FBAS
from test_utils import get_test_data_file_path

def benchmark():
    # file = 'almost_symmetric_network_12_orgs.json'
    file = 'almost_symmetric_network_5_orgs_delete_prob_factor_1.json'
    fbas = FBAS.from_stellarbeat_json(get_validators_from_file(get_test_data_file_path(file)))
    t = timeit(lambda: optimal_overlay(fbas), number=1)
    print(t)

if __name__ == '__main__':
    benchmark()