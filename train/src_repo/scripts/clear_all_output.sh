#!/bin/bash

PROJECT_BASE="/project/train"
SRC_REPO="${PROJECT_BASE}/src_repo"

if [ -e "${SRC_REPO}/infer_test_output" ]; then
  rm -r "${SRC_REPO}/infer_test_output"
fi

if [ -e "${SRC_REPO}/datasets" ]; then
  rm -r "${SRC_REPO}/datasets"
fi

if [ -e "${PROJECT_BASE}/models" ]; then
  rm -r "${PROJECT_BASE}/models"
fi

if [ -e "${PROJECT_BASE}/tensorboard" ]; then
  rm -r "${PROJECT_BASE}/tensorboard"
fi
