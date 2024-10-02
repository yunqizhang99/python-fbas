from python_fbas.fbas_generator import gen_symmetric_fbas

def test_gen_symmetric_fbas():
    fbas = gen_symmetric_fbas(3)
    assert len(fbas.qset_map) == 9