#!/bin/bash

git clone https://github.com/BorealisAI/advertorch
cd advertorch
python setup.py install
cd ..
rm -rf advertorch
pip install -r requirements.txt
