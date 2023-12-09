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
- 包含前两个py脚本的调用

推理测试
- `ji.py`
  - 同样基于模块`postprocess.py`
  - `project/train`中文件修改后，需要把`ev_sdk`中所有的同名脚本文件进行更新

旧：*当前方案：三分类模型直接拼一个栏杆mask模型，两个模型独立推理，结果合并成json格式输出*

当前方案：利用上面输出的json，在经过一个二分类classifier，根据classifier的输出，修改对应位置json内容

linux上运行的项目，改成windows下运行需要修改哪些地方：
- scripts脚本，bat脚本换成sh脚本，并且设置project根路径
- `config.py`设置project根路径
- `yaml_files`修改引用路径
- 部署的时候，跑一遍`pipeline.sh`看看哪里报错


## 备忘录

编码阶段需要注意的内容
- 减小`train.py`的epoch数
- 跑一遍`bash /project/train/src_repo/scripts/pipeline.sh`，和
- 如果出问题，先在本地项目上debug，然后同步所有修改文件到极市平台

训练之前需要注意的内容
- 增加epoch数
- 控制batch_size，使得不会被kill
- 修改net_config，二分类的训练设置参数
- 使用一下`bash /project/train/src_repo/scripts/clear_all_output.sh`

git更新：
- `git add train ev_sdk README.md`
- 适时增加`gitignore`内容

提交训练脚本：`bash /project/train/src_repo/scripts/for_training_on_jishi.sh`
测试无需脚本


## TODO
///////// 
用于测试的文件结构需要修改一下——有bug，总是改不对，感觉是自己测试平台账号的问题……
？ONNX+TensorRT？