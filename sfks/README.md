# CAIL2020——司法考试

该项目为 **CAIL2020——司法考试** 的代码和模型提交说明。

数据集下载请访问比赛[主页](http://cail.cipsc.org.cn/)。

## 数据集引用

如果你要在学术论文中引用数据集，请使用如下bib

```tex
@article{zhong2019jec,
  title={JEC-QA: A Legal-Domain Question Answering Dataset},
  author={Zhong, Haoxi and Xiao, Chaojun and Tu, Cunchao and Zhang, Tianyang and Liu, Zhiyuan and Sun, Maosong},
  journal={arXiv preprint arXiv:1911.12011},
  year={2019}
}
```

## 选手交流群

QQ群：237633234

## 数据说明

本任务所使用的数据集来自于论文``JEC-QA: A Legal-Domain Question Answering Dataset``的司法考试数据集。

下发的文件包含``0_train.json,1_train.json``，分别对应概念理解题和情景分析题。

两个文件均包含若干行，每行数据均为json格式，包含若干字段：

- ``answer``：代表该题的答案。
- ``id``：题目的唯一标识符。
- ``option_list``：题目每个选项的描述。
- ``statement``：题干的描述。
- ``subject``：代表该问题所属的分类，仅有部分数据含有该字段。
- ``type``：无意义字段。

实际测试数据不包含``answer``字段。

## 提交的文件格式及组织形式

你可以在 ``python_sample`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``python_sample/main.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``/input/``中读取数据进行预测。

在该文件夹中包含**若干**文件，每个文件均由若干行``json``格式数据组成。每行的数据格式与下发数据格式完全一致。选手需要从将预测的结果输出到``/output/result.txt``中，以``json``格式输出一个字典。对于编号为``id``的题目，你需要在输出的字典中设置``id``字段，并且该字段内容为该题答案，类型为``list``。

以上为 ``main.py`` 中你需要实现的内容，你可以利用 ``python_example`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/work``路径下然后运行。

**请注意，如果你想要自己通过命令行运行python代码，请按照如下命令运行**

```bash
sudo /home/user/miniconda/bin/python3 work.py
```

## 其他语言的支持

如上文所述，我们现阶段只支持 ``python`` 语言的提交，但是这并不代表你不能够使用其他语言进行预测。你可以使用``python3 main.py``去调用运行其他语言的命令。但请注意，在你调用其他命令的时候请在命令前加上``sudo``以保证权限不会出问题。

## 现有的系统环境

| 软件名称 | 版本号 |
| -------- | ------ |
| python   | 3.6.9  |
| g++      | 5.4.0  |
| gcc      | 5.4.0  |

python库的环境列表：

```
Package                          Version            
-------------------------------- -------------------
absl-py                          0.9.0              
anykeystore                      0.2                
apex                             0.9.10.dev0        
asn1crypto                       1.3.0              
astor                            0.8.1              
attrs                            19.3.0             
backcall                         0.1.0              
beautifulsoup4                   4.9.0              
bert-serving-client              1.10.0             
bert-serving-server              1.10.0             
bleach                           3.1.5              
blis                             0.4.1              
boto                             2.49.0             
boto3                            1.13.3             
botocore                         1.16.3             
Bottleneck                       1.3.2              
bz2file                          0.98               
cachetools                       4.1.0              
catalogue                        1.0.0              
certifi                          2020.4.5.1         
cffi                             1.14.0             
chardet                          3.0.4              
click                            7.1.2              
conda                            4.8.3              
conda-package-handling           1.6.0              
cryptacular                      1.5.5              
cryptography                     2.8                
cycler                           0.10.0             
cymem                            2.0.3              
Cython                           0.29.17            
dataclasses                      0.7                
decorator                        4.4.2              
defusedxml                       0.6.0              
docutils                         0.15.2             
fastai                           1.0.61             
fastprogress                     0.2.3              
fasttext                         0.9.2              
Flask                            1.1.2              
gast                             0.2.2              
gensim                           3.8.3              
google-auth                      1.14.1             
google-auth-oauthlib             0.4.1              
google-pasta                     0.2.0              
GPUtil                           1.4.0              
grpcio                           1.28.1             
h5py                             2.10.0             
html5lib                         1.0.1              
hupper                           1.10.2             
idna                             2.9                
importlib-metadata               1.6.0              
ipython                          7.14.0             
ipython-genutils                 0.2.0              
itsdangerous                     1.1.0              
jedi                             0.17.0             
jeepney                          0.4.3              
jieba                            0.42.1             
Jinja2                           2.11.2             
jmespath                         0.9.5              
joblib                           0.14.1             
JPype1                           0.7.0              
jsonschema                       3.2.0              
Keras                            2.3.1              
Keras-Applications               1.0.8              
keras-bert                       0.81.0             
keras-embed-sim                  0.7.0              
keras-layer-normalization        0.14.0             
keras-multi-head                 0.22.0             
keras-pos-embd                   0.11.0             
keras-position-wise-feed-forward 0.6.0              
Keras-Preprocessing              1.1.0              
keras-self-attention             0.41.0             
keras-transformer                0.33.0             
keyring                          21.2.1             
keyrings.alt                     3.4.0              
kiwisolver                       1.2.0              
lda                              1.1.0              
lightgbm                         2.3.1              
Mako                             1.1.2              
Markdown                         3.2.1              
MarkupSafe                       1.1.1              
matplotlib                       3.2.1              
mkl-fft                          1.0.15             
mkl-random                       1.1.0              
mkl-service                      2.3.0              
mock                             4.0.2              
murmurhash                       1.0.2              
ninja                            1.9.0.post1        
nltk                             3.5                
numexpr                          2.7.1              
numpy                            1.18.1             
nvidia-ml-py3                    7.352.0            
oauthlib                         3.1.0              
olefile                          0.46               
opt-einsum                       3.2.1              
packaging                        20.3               
pandas                           1.0.3              
parso                            0.7.0              
PasteDeploy                      2.1.0              
pbkdf2                           1.3                
pbr                              3.1.1              
pexpect                          4.8.0              
pickleshare                      0.7.5              
Pillow                           7.0.0              
pip                              20.0.2             
plac                             1.1.3              
plaster                          1.0                
plaster-pastedeploy              0.7                
preshed                          3.0.2              
prompt-toolkit                   3.0.5              
protobuf                         3.11.3             
ptyprocess                       0.6.0              
pyasn1                           0.4.8              
pyasn1-modules                   0.2.8              
pybind11                         2.5.0              
pycosat                          0.6.3              
pycparser                        2.20               
pycrypto                         2.6.1              
pycurl                           7.43.0.5           
Pygments                         2.6.1              
pyhanlp                          0.1.64             
pyOpenSSL                        19.1.0             
pyparsing                        2.4.7              
pyramid                          1.10.4             
pyramid-mailer                   0.15.1             
pyrsistent                       0.16.0             
PySocks                          1.7.1              
python-dateutil                  2.8.1              
python3-openid                   3.1.0              
pytorch-pretrained-bert          0.6.2              
pytorch-transformers             1.2.0              
pytz                             2020.1             
pyxdg                            0.26               
PyYAML                           5.3.1              
pyzmq                            19.0.1             
regex                            2020.4.4           
repoze.sendmail                  4.4.1              
requests                         2.23.0             
requests-oauthlib                1.3.0              
rsa                              4.0                
ruamel-yaml                      0.15.87            
s3transfer                       0.3.3              
sacremoses                       0.0.43             
scikit-learn                     0.22.2.post1       
scikit-multilearn                0.2.0              
scipy                            1.4.1              
SecretStorage                    3.1.2              
sentencepiece                    0.1.86             
setuptools                       46.1.3.post20200330
simplegeneric                    0.8.1              
six                              1.14.0             
sklearn                          0.0                
smart-open                       2.0.0              
soupsieve                        2.0                
spacy                            2.2.4              
SQLAlchemy                       1.3.16             
srsly                            1.0.2              
taboo                            0.8.8              
tensorboard                      2.1.1              
tensorflow                       2.1.0              
tensorflow-estimator             2.1.0              
tensorflow-hub                   0.8.0              
termcolor                        1.1.0              
tflearn                          0.3.2              
Theano                           1.0.4              
thinc                            7.4.0              
thulac                           0.2.1              
torch                            1.4.0              
torchvision                      0.5.0              
tqdm                             4.46.0             
traitlets                        4.3.3              
transaction                      3.0.0              
translationstring                1.3                
typing                           3.7.4.1            
urllib3                          1.25.8             
velruse                          1.1.1              
venusian                         3.0.0              
wasabi                           0.6.0              
wcwidth                          0.1.9              
webencodings                     0.5.1              
WebOb                            1.8.6              
Werkzeug                         1.0.1              
wheel                            0.34.2             
wrapt                            1.12.1             
WTForms                          2.3.1              
wtforms-recaptcha                0.3.2              
xgboost                          1.0.2              
zipp                             3.1.0              
zope.deprecation                 4.4.0              
zope.interface                   5.1.0              
zope.sqlalchemy                  1.3
filelock                         3.0.12
tokenizers                       0.7.0
transformers                     2.9.1
```

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。