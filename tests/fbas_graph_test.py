import logging
import pytest
from test_utils import get_test_data_list, get_validators_from_test_fbas
from python_fbas.fbas_graph import FBASGraph

def test_collapse():
    data = get_test_data_list()
    for f,d in data.items():
        fg = FBASGraph.from_json(d)
        fg.check_integrity()
        logging.info("graph of %s before flattening:\n %s", f, fg.stats())
        fg.flatten_diamonds()
        logging.info("graph of %s after flattening:\n %s", f, fg.stats())

def test_is_quorum():
    fbas1 = FBASGraph.from_json(get_validators_from_test_fbas('conflicted.json'))
    assert fbas1.is_quorum({'PK11','PK12','PK13'})
    assert fbas1.is_quorum({'PK11','PK12'})
    assert not fbas1.is_quorum({'PK11'})
    assert not fbas1.is_quorum({'PK11','PK12','PK13','PK21'})
    assert not fbas1.is_quorum({'PK13','PK21'})
    assert fbas1.is_quorum({'PK11','PK12','PK13','PK21','PK22','PK23'})
    with pytest.raises(AssertionError):
        fbas1.is_quorum({'PK11','PK12','PK13','NON_EXISTENT'})
    assert not fbas1.is_quorum({'PK11','PK12','PK13','PKX'})
    assert fbas1.is_quorum({'PK11','PK12','PKX','PK22','PK23'})
    assert not fbas1.is_quorum({'PK11','PK12','PKX','PK22'})