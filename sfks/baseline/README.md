项目使用方法请参考[这里](https://github.com/haoxizhong/pytorch-worker)。

该模型为用Attention整合题干和选项，然后进行预测的模型。

训练步骤：

1. 将训练数据拷贝至``data/data``下并根据``config/model.config``划分训练集和验证集。
2. 使用``utils/cutter.py``进行分词操作
3. 使用``python3 train.py --config config/model.config --gpu 0``进行训练。

提交步骤：

​	参考``main.py``。