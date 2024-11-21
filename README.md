# python-fbas

## Usage

Optionally create a virtual environment:

```
python3 -m venv venv
source venv/bin/activate
```

Install requirements:
```
pip install -r requirements.txt
```

Run the tests:
```
python3 -m pytest
```

Run:
```
python3 main.py
```

To check whether the current Stellar network has quorum intersection, first download the latest data from stellarbeat:
```
python3 main.py update-stellarbeat-cache
```
Then, check intersection with the fast heuristic (which may return `'unknown'`) using the new codebase:
```
python3 main.py --log-leve=INFO --new --fbas=stellarbeat check-intersection --fast
```
To check intersection using a SAT solver (always returns yes or no):
```
python3 main.py --log-leve=INFO --new --totalizer --fbas=stellarbeat check-intersection --cnf
```
There are other encodings but they are slower (try `--z3` and `--pysat_fmla`)

To determine the minimal number of nodes that, if corrupted, can split the network:
```
python3 main.py --log-leve=INFO --new --totalizer --fbas=stellarbeat min-splitting-set
```

You can also provide a FBAS to check in JSON format:
```
python3 main.py --log-leve=INFO --new --totalizer --fbas=tests/test_data/random/almost_symmetric_network_16_orgs_delete_prob_factor_1.json check-intersection --cnf
```
