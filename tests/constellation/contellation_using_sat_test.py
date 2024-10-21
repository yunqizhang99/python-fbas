import logging
from python_fbas.constellation.constellation_using_sat import constellation_graph
from python_fbas.fbas_generator import gen_symmetric_fbas

logging.basicConfig(level=logging.INFO)

def test_constellation_1():
    fbas = gen_symmetric_fbas(3)
    g = constellation_graph(fbas)
    logging.info(g)