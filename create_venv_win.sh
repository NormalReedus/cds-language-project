#!/usr/bin/env bash

VENVNAME=lang101

python -m venv $VENVNAME
source $VENVNAME/Scripts/activate
python get-pip.py

# delete after testing
pip install ipython # remove after testing
pip install jupyter # remove after testing 

python -m ipykernel install --user --name=$VENVNAME # remove after testing

test -f requirements.txt && pip install -r requirements.txt

# Makes sure the required directories are present (since git does not clone empty folders)
mkdir -p data/
mkdir -p output/

echo "build $VENVNAME"