# CAIL2020——论辩挖掘

该项目为 **CAIL2020——论辩挖掘** 的代码和模型提交说明。

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。

## 选手交流群

QQ群：237633234

## 论辩挖掘及其研究方向

论辩挖掘(Argumentation Mining)是自然语言处理领域一个新颖的研究方向，与本次测评相关的两份研究工作可参考：

https://www.aclweb.org/anthology/C18-1314.pdf

https://arxiv.org/pdf/1911.01621

## 数据说明

本任务第一阶段所下发的文件包含``SMP-CAIL2020-train.csv``,`` SMP-CAIL2020-text.csv``，分别包含以下内容：

1. ``SMP-CAIL2020-text-train.csv``：包含了裁判文书所有对于辩诉双方辩护全文的数据。分别包含以下维度：
				``sentence_id``： 句子id
		``text_id``： 裁判文书id
		``position``： 二分类标签：sc——诉方；bc——辩方
		``sentence``： 句子文本
	
2. ``SMP-CAIL2020-train.csv``： 包含了2449对裁判文书中的互动论点对。分别包含以下维度：

  ​	``id``： 论点对id
  ​	``text_id``： 裁判文书id
  ​	``sc``： 论点对中诉方论点
  ​	``A/B/C/D/E``： 给出的五句候选辩方论点
  ​	``answer``： 辩方正确论点

实际测试数据与``SMP-CAIL2020-train.csv``格式大部分相同，但不包含``answer``字段。

## 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_sample/main.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/``中读取数据进行预测。

第一阶段测试中，该文件夹包含**两个**文件，``SMP-CAIL2020-text-test1.csv, SMP-CAIL2020-test1.csv``两个文件中，``text``数据同下发数据中的``SMP-CAIL2020-text-train.csv``格式完全一致；``SMP-CAIL2020-test1.csv``与同下发数据中的``SMP-CAIL2020-train.csv``格式基本一致，但缺少选手``answer``维度数据。选手需要从将预测的结果输出到``/output/result1.csv``中，以``csv``格式输出。输出文件中必须包含且只包含``id``和``answer``两个维度，可参考下发文件夹中``Sample_submission.csv``。

以上为 ``main.py`` 中你需要实现的内容，你可以利用 ``python_example`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/work``路径下然后运行。

## 其他语言的支持

如上文所述，我们现阶段只支持 ``python`` 语言的提交，但是这并不代表你不能够使用其他语言进行预测。你可以使用``python3 main.py``去调用运行其他语言的命令。但请注意，在你调用其他命令的时候请在命令前加上``sudo``以保证权限不会出问题。

## 基线模型

本次测评共提供了基于随机模型和基于神经网络模型各两种的基准线模型。

- 随机模型：(代码见/python_sample)
  - 全选A——模型将全部选择第一个选项。
  - 公共词——模型基于诉称以及辩称五句候选中文本的分词重叠程度，选出重叠最多的作为答案。
- 神经网络模型：(代码见https://github.com/gaoyixu/CAIL2020-Argument-Mining)
  - LSTM
  - BERT

本次评测将以BERT模型给出的准确率作为第一阶段的基准线。

## 现有的系统环境

| 软件名称 | 版本号 |
| -------- | :----- |
| python   | 3.6.6  |
| g++      | 5.4.0  |
| gcc      | 5.4.0  |

python库的环境列表：

```
Package                 Version        Status
----------------------- -------------- ----------------------
Cython          		- 0.29         - active
Jinja2          		- 2.10         - active
Keras-Applications 		- 1.0.6        - active
Keras-Preprocessing 	- 1.0.5        - active
Keras           		- 2.2.4        - active
Mako            		- 1.0.7        - active
Markdown        		- 3.0.1        - active
MarkupSafe      		- 1.0          - active
Pillow          		- 5.3.0        - non-active
Pillow          		- 5.4.1        - active
PyWavelets      		- 1.0.1        - active
PyYAML          		- 3.13         - non-active
PyYAML          		- 3.13         - non-active
PyYAML          		- 3.13         - active
PyYAML          		- 3.13         - active
Pygments        		- 2.2.0        - active
Send2Trash      		- 1.5.0        - active
Theano          		- 1.0.3+2.g3e47d39ac.dirty - active
Werkzeug        		- 0.14.1       - active
absl-py         		- 0.6.1        - active
astor           		- 0.7.1        - active
backcall        		- 0.1.0        - active
bleach          		- 3.0.2        - active
certifi         		- 2018.10.15   - active
cffi            		- 1.11.5       - non-active
cffi            		- 1.12.1       - active
chainer         		- 4.5.0        - active
chardet         		- 3.0.4        - active
cloudpickle     		- 0.6.1        - non-active
cloudpickle     		- 0.8.0        - active
cntk-gpu        		- 2.6          - active
cycler          		- 0.10.0       - non-active
cycler          		- 0.10.0       - non-active
cycler          		- 0.10.0       - active
cycler          		- 0.10.0       - active
cytoolz         		- 0.9.0.1      - active
dask            		- 0.20.0       - non-active
dask            		- 1.1.2        - active
decorator       		- 4.3.0        - active
entrypoints     		- 0.2.3        - active
filelock        		- 3.0.10       - active
gast            		- 0.2.0        - active
get             		- 1.0.3        - active
graphviz        		- 0.8.4        - active
grpcio          		- 1.16.1       - active
h5py            		- 2.8.0        - non-active
h5py            		- 2.8.0        - non-active
h5py            		- 2.8.0        - active
h5py            		- 2.8.0        - active
idna            		- 2.6          - active
imageio         		- 2.5.0        - active
ipykernel       		- 5.1.0        - active
ipython_genutils 		- 0.2.0        - active
ipython         		- 7.1.1        - active
ipywidgets      		- 7.4.2        - active
jedi            		- 0.13.1       - active
jsonschema      		- 2.6.0        - active
jupyter-client  		- 5.2.3        - active
jupyter-console 		- 6.0.0        - active
jupyter-core    		- 4.4.0        - active
jupyter         		- 1.0.0        - active
kiwisolver      		- 1.0.1        - non-active
kiwisolver      		- 1.0.1        - non-active
kiwisolver      		- 1.0.1        - active
kiwisolver      		- 1.0.1        - active
leveldb         		- 0.20         - active
lightgbm        		- 2.2.1        - active
matplotlib      		- 3.0.0        - non-active
matplotlib      		- 3.0.2        - active
mistune         		- 0.8.4        - active
mkl-fft         		- 1.0.10       - active
mkl-random      		- 1.0.2        - active
mxnet-cu90      		- 1.3.0.post0  - active
nbconvert       		- 5.3.1        - active
nbformat        		- 4.4.0        - active
networkx        		- 2.2          - non-active
networkx        		- 2.2          - non-active
networkx        		- 2.2          - active
networkx        		- 2.2          - active
nltk            		- 3.4.5        - active
notebook        		- 5.6.0        - active
numpy           		- 1.15.4       - active
olefile         		- 0.46         - active
opencv-python   		- 3.4.3.18     - active
pandas          		- 0.23.4       - non-active
pandas          		- 0.24.1       - active
pandocfilters   		- 1.4.2        - active
parso           		- 0.3.1        - active
pexpect         		- 4.6.0        - active
pickleshare     		- 0.7.5        - active
pip             		- 18.1         - active
post            		- 1.0.2        - active
prometheus-client 		- 0.4.2        - active
prompt-toolkit  		- 2.0.7        - active
protobuf        		- 3.6.1        - active
ptyprocess      		- 0.6.0        - active
public          		- 2.0.1        - active
pycorenlp       		- 0.3.0        - active
pycparser       		- 2.19         - non-active
pycparser       		- 2.19         - non-active
pycparser       		- 2.19         - active
pycparser       		- 2.19         - active
pydot           		- 1.2.4        - active
pygpu           		- 0.7.6        - active
pyparsing       		- 2.2.2        - non-active
pyparsing       		- 2.3.1        - active
python-dateutil 		- 2.7.5        - active
python-gflags   		- 3.1.2        - active
pytz            		- 2018.5       - non-active
pytz            		- 2018.9       - active
pyzmq           		- 17.1.2       - active
qtconsole       		- 4.4.2        - active
query-string    		- 1.0.2        - active
request         		- 1.0.2        - active
requests        		- 2.18.4       - active
scikit-image    		- 0.14.1       - non-active
scikit-image    		- 0.14.1       - non-active
scikit-image    		- 0.14.1       - active
scikit-image    		- 0.14.1       - active
scikit-learn    		- 0.20.0       - active
scipy           		- 1.2.1        - active
setuptools      		- 40.5.0       - active
six             		- 1.11.0       - active
sklearn         		- 0.0          - active
tensorboard     		- 1.12.0       - active
tensorflow-gpu  		- 1.12.0       - active
termcolor       		- 1.1.0        - active
terminado       		- 0.8.1        - active
testpath        		- 0.4.2        - active
tflearn         		- 0.3.2        - active
toolz           		- 0.9.0        - non-active
toolz           		- 0.9.0        - non-active
toolz           		- 0.9.0        - active
toolz           		- 0.9.0        - active
torch           		- 0.4.1        - non-active
torch           		- 1.0.0        - active
torchfile       		- 0.1.0        - active
torchnet        		- 0.0.4        - active
torchvision     		- 0.2.1        - non-active
tornado         		- 5.1.1        - active
tqdm            		- 4.28.1       - active
traitlets       		- 4.3.2        - active
urllib3         		- 1.22         - active
visdom          		- 0.1.8.5      - active
wcwidth         		- 0.1.7        - active
webencodings    		- 0.5.1        - active
websocket-client 		- 0.53.0       - active
wheel           		- 0.32.2       - active
widgetsnbextension 		- 3.4.2        - active
xgboost         		- 0.80         - active
yolk            		- 0.4.3        - active
yolk3k          		- 0.9          - active
```

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。