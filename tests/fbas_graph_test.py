import logging
from test_utils import get_test_data_list
from python_fbas.fbas_graph import FBASGraph


def test_collapse():
    data = get_test_data_list()
    for d in data:
        fg = FBASGraph.from_json(d)
        fg.check_integrity()
        logging.info("stats before collapse: %s", fg.stats())
        fg.collapse()
        logging.info("stats after collapse: %s", fg.stats())