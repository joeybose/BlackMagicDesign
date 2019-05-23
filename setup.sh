#!/bin/bash
root_dir=$(pwd)

# Install advertorch
git clone https://github.com/BorealisAI/advertorch
cd advertorch
python setup.py install
cd ..
rm -rf advertorch

# Install glove
cd $root_dir
cd setup_scripts/
bash download_glove.sh

## Install data
cd $root_dir
cd setup_scripts/
bash download_data.sh

## Download pretrained LSTM
cd $root_dir
cd setup_scripts/
bash download_lstm.sh
