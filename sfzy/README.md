# CAIL2020——司法摘要

该项目为 **CAIL2020—司法摘要** 的代码和模型提交说明。

## 选手交流群

QQ群：237633234

## 数据说明

本任务技术评测使用的训练集、验证集、测试集来自于司法大数据院提供的法院裁判文书，约10000篇裁判文书以及对应的司法摘要，其中所涉及到的裁判文书均为民事一审判决书。

训练数据均包含若干行，每行数据为json格式，包含若干字段：

* ``id``：样本唯一标识符。
* ``summary``：样本的摘要内容。
* ``text``：将原裁判文书按照句子切分开，内含2个字段。
    * ``sentence``：表示句子的内容。
    * ``label``：表示句子的重要度。

实际测试数据不包含``summary``字段和``text``中的``label``。


## 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_sample/main.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python main.py``或``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/input.json``中读取数据进行预测以得到相应摘要，该数据格式与下发数据格式完全一致，但会删除 "label"、"summary"字段。选手需要将预测的结果输出到`/output/result.json`中，预测结果文件为一个json格式的文件，具体可以查看 ``evaluate/result.json``。


## 评测脚本

本赛道采用的ROUGE(Recall-Oriented Understudy for Gisting Evaluation)评价评价。ROUGE指标将自动生成的摘要与参考摘要进行比较, 其中ROUGE-1衡量unigram匹配情况，ROUGE-2衡量bigram匹配，ROUGE-L记录最长的公共子序列。三者都只采用f-score，且总分的计算方式为：```0.2*f-score(R1)+0.4*f-score(R2)+0.4*f-score(RL)```

我们在 ``evaluate`` 文件夹中摘要生成的样例

## 现有的系统环境

[pytorch1.5.1](./envs/pytorch1.5.1.md)

[tf2.0](./envs/tf2.0.md)

[tf1.14](./envs/tf1.14.md)

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。
