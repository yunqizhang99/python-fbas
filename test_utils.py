import os

def get_test_data_file_path(name):
    return os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 'test_data', name)