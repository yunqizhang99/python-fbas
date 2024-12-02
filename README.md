# python-fbas

## Usage

Optionally create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install the package:
```
pip install .
```

Run the tests:
```
pip install pytest
python3 -m pytest
```

Run the main script and obtain the help message:
```
python-fbas
```

To check whether the current Stellar network has quorum intersection:
```
python-fbas --log-leve=INFO --fbas=stellarbeat check-intersection
```

To determine the minimal number of nodes that, if corrupted, can split the network:
```
python-fbas --log-leve=INFO --fbas=stellarbeat min-splitting-set
```
To determine the minimal number of nodes that, if corrupted, can halt the network:
```
python-fbas --log-leve=INFO --fbas=stellarbeat min-blocking-set
```

You can also provide a FBAS to check in JSON format:
```
python-fbas --log-leve=INFO --fbas=tests/test_data/random/almost_symmetric_network_16_orgs_delete_prob_factor_1.json check-intersection
```

Note that stellarbeat data is cached in a file the first time it is needed.
To update the cache:
```
python-fbas update-stellarbeat-cache
```

