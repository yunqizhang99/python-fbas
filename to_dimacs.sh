#!/bin/bash

VENV=.venv
if [ ! -d "$VENV" ]; then
    echo "Virtual environment not found. Please install python-fbas in a virtual environment first (and adjust the VENV variable in this script)."
    exit 1
fi

source $VENV/bin/activate

find tests/test_data/random/ -type f ! -name "*orgs.json" ! -name "*core.json" -name "*.json" | while read -r file; do
    output_file="${file%.json}.dimacs"
    python-fbas --cardinality-encoding=naive --output-problem="$output_file" --fbas="$file" check-intersection
done

deactivate
