import logging
from python_fbas.constellation.constellation import *

def test_regular_fbas_to_fbas_graph():
    regular_fbas_to_fbas_graph({'A':(1, ['A'])})
    regular_fbas_to_fbas_graph({'A':(2, ['A','B','C']), 'B':(2, ['A','B']), 'C':(1, ['A','B','C'])})

def test_compute_clusters():
    # regular fbas with 10 organizations, each with threshold 7:
    reqs = (7, [f"O_{i}" for i in range(1,11)])
    regular_fbas = {f"O_{i}":reqs for i in range(1,11)}
    clusters = compute_clusters(regular_fbas)
    logging.info("clusters: %s", clusters)
    assert clusters ==  [{'O_3', 'O_5', 'O_1', 'O_2', 'O_4'}, {'O_10', 'O_9', 'O_7', 'O_6', 'O_8'}]
    # regular fbas with 11 organizations, each with threshold 8:
    reqs = (8, [f"O_{i}" for i in range(1,12)])
    regular_fbas = {f"O_{i}":reqs for i in range(1,12)}
    clusters = compute_clusters(regular_fbas)
    logging.info("clusters: %s", clusters)
    assert clusters == [{'O_1', 'O_2', 'O_3', 'O_4'}, {'O_7', 'O_6', 'O_8', 'O_5'}, {'O_10', 'O_11', 'O_9'}]
    # non-uniform thresholds:
    orgs = [f"O_{i}" for i in range(1,12)]
    t1 = 9
    t2 = 6
    t3 = 8
    regular_fbas = {f"O_{i}":(t1, orgs) for i in range(1,4)}\
        | {f"O_{i}":(t2, orgs) for i in range(4,8)} | {f"O_{i}":(t3, orgs) for i in range(8,12)}
    clusters = compute_clusters(regular_fbas)
    logging.info("clusters: %s", clusters)
    assert clusters == [{'O_7', 'O_6', 'O_5', 'O_4'}, {'O_9', 'O_11', 'O_8', 'O_10'}, {'O_2', 'O_1', 'O_3'}]