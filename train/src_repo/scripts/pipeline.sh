#!/bin/bash

PROJECT_BASE="/project/train"

cd $PROJECT_BASE/src_repo

python preprocess.py
python train.py
python infer_test.py