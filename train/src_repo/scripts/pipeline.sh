#!/bin/bash

PROJECT_BASE="/C/Users/Dusk_Hermit/Desktop/test/tensorflow_test/train"

# 全流程

# 数据集处理
python "$(dirname "$0")/preprocess.py"

# 训练
"$(dirname "$0")/train.sh"

# 推理测试
"$(dirname "$0")/predict_testing.sh"
