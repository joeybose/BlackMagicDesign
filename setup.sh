#!/bin/bash
root_dir=$(pwd)

echo "Install advertorch\n"
git clone https://github.com/BorealisAI/advertorch
cd advertorch
python setup.py install
cd ..
rm -rf advertorch

echo "Installing python packages\n"
pip install -r requirements.txt

echo "Installing glove\n"
cd $root_dir
cd setup_scripts/
bash download_glove.sh

echo "Installing data\n"
cd $root_dir
cd setup_scripts/
bash download_data.sh

echo "Downloading pretrained LSTM\n"
cd $root_dir
cd setup_scripts/
bash download_lstm.sh
