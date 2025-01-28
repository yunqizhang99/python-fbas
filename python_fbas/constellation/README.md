# Constellation

In this package we provide functionality related to Constellation overlays.

Constellation is described in an upcoming FC 2025 paper.

## Installation

Start in the top-level directory (i.e., the grand-parent of the directory this file belongs to).
Create a virtual environment and activate it:
```
python3 -m venv venv
source venv/bin/activate
```
Next, run `pip install .` (or `pip install -e .` to keep updated during development).
Finally, go to `python_fbas/constellation/brute-force-search/` and run `make install` (note this calls the gcc compiler).

## Examples

To compute the Constellation clusters when there are 200 organizations where 100 have threshold 130 and 100 have threshold 140, try
```
constellation compute-clusters --thresholds 100 130 100 140 --max-num-clusters=5 --min-cluster-size=33
```
