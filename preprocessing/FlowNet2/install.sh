#!/bin/bash
cd ./networks/correlation_package
rm -rf *_cuda.egg-info build dist __pycache__
pip install .

cd ../resample2d_package
rm -rf *_cuda.egg-info build dist __pycache__
pip install .

cd ../channelnorm_package
rm -rf *_cuda.egg-info build dist __pycache__
pip install .

cd ..
