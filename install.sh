#!/bin/bash 

cd ./packages/resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
pip install .

cd ../../
