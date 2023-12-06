## 项目结构


训练过程已在极市编码平台跑通，无错

编码阶段全流程脚本：`train/src_repo/scripts/pipeline.sh`
- `preprocess.py`
  - 输出三个yolo模型需要的训练数据集
- `train.py`
  - 三个yolo模型轮流训练
  - 自动保存最新的训练结果到`/project/train/tensorboard`中
  - 自动寻找最新版的`last.pt`并在此基础上训练
- `infer_test.py`
  - 能测试看看输出最终需要的json格式+mask图片
  - 自动寻找最新版的`best.pt`并使用它推理
  - 基于模块`postprocess.py`

极市平台训练：`bash /project/train/src_repo/scripts/for_training_on_jishi.sh`

推理测试（还未做）
- `ji.py`
  - 同样基于模块`postprocess.py`
  - `project/train`中文件修改后，需要把`ev_sdk`中有的同名脚本文件进行更新

当前方案：三分类模型直接拼一个栏杆mask模型，两个模型独立推理，结果合并成json格式输出

todo
- 新的架构，如yoloseg和yolo-class1-detect的输出结果，最后加一个二分类classifier，进行判别
  - 这就需要改很多地方了……
  - // 尽量新的架构不要变化太多，能使用已经训练的模型最好
- 其他竞赛可用的技术？冲浪一下打开视野？
- // 说实话时间不够就是万策尽，怎么救））
