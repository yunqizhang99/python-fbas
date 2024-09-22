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

## Compute a Constellation overlay

To compute an overlay roughly following the Constellation algorithm for the current (9.22.2024) Top Tier:

```
python3 main.py --fbas=tests/test_data/top_tier.json constellation-overlay

```

This will print the list of Constellation clusters (after about 2 minutes on my desktop machine).
