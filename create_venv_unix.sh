#!/usr/bin/env bash

VENVNAME=lang101

python3 -m venv $VENVNAME
source $VENVNAME/bin/activate
pip install --upgrade pip

test -f requirements.txt && pip install -r requirements.txt

mkdir -p data/
mkdir -p output/

echo "build $VENVNAME"