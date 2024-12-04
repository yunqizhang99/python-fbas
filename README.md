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

To check whether the current Stellar network, according to data from [stellarbeat.io](https://stellarbeat.io), has quorum intersection:
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

For the `min-splitting-set` and `min-blocking-set` commands, you can group validators by attribute.
For example:
```
python-fbas --log-leve=INFO --fbas=stellarbeat --group-by=homeDomain min-splitting-set
```
This computes the minimal number of home domains that must be corrupted in order to create disjoint quorums.

You might get surprising results due to a single validators having a weird configuration, and you might not care about this problematic validator.
In this case it helps to restrict the analysis to the validators that are reachable from some validator you care about.
For example, to restrict the FBAS to what is reachable from one of SDF's validators:
```
python-fbas --log-level=INFO --group-by=homeDomain --validator=GCGB2S2KGYARPVIA37HYZXVRM2YZUEXA6S33ZU5BUDC6THSB62LZSTYH min-splitting-set
```

Finally, you can also provide a FBAS to check in JSON format:
```
python-fbas --log-leve=INFO --fbas=tests/test_data/random/almost_symmetric_network_13_orgs_delete_prob_factor_1.json check-intersection
```

Note that data form [stellarbeat.io](https://stellarbeat.io) is cached in a local file the first time it is needed.
To update the cache:
```
python-fbas update-stellarbeat-cache
```

