from timeit import timeit
from pysat.solvers import SolverNames
from python_fbas.sat_based_fbas_analysis import check_intersection
from python_fbas.fbas import FBAS
from tests.test_utils import get_validators_from_test_data_file

# get all the solvers available:
solvers = {f : getattr(SolverNames,f)[0] for f in dir(SolverNames) if not f.startswith('__')}

def benchmark_intersection_checker():
    file = 'almost_symmetric_network_16_orgs_delete_prob_factor_4.json'
    fbas = FBAS.from_json(get_validators_from_test_data_file(file))
    for name, shortname in solvers.items():
        print(name)
        t = timeit(lambda: check_intersection(fbas, solver=shortname), number=1)
        print(t)

if __name__ == '__main__':
    benchmark_intersection_checker()