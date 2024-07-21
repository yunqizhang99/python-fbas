from timeit import *
from sat_based_fbas_analysis import check_intersection
from test_utils import get_validators_from_file
from fbas import FBAS
from pysat.solvers import SolverNames
from test_utils import get_test_data_file_path

# get all the solvers available:
solvers = {f : getattr(SolverNames,f)[0] for f in dir(SolverNames) if not f.startswith('__')}

def benchmark_intersection_checker():
    file = 'almost_symmetric_network_16_orgs_delete_prob_factor_4.json'
    fbas = FBAS.from_stellarbeat_json(get_validators_from_file(get_test_data_file_path(file)))
    for name, shortname in solvers.items():
        print(name)
        t = timeit(lambda: check_intersection(fbas, solver=shortname), number=1)
        print(t)

if __name__ == '__main__':
    benchmark_intersection_checker()