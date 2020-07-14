# CAIL2020——阅读理解

该项目为 **CAIL2020—阅读理解** 的代码和模型提交说明。

## 选手交流群

QQ群：237633234

## 数据说明

本任务技术评测训练集包括两部分，一部分为去年的[CJRC训练集](https://github.com/china-ai-law-challenge/CAIL2019/tree/master/%E9%98%85%E8%AF%BB%E7%90%86%E8%A7%A3/data). 一部分为重新标注的约5100个问答对，其中民事、刑事、行政各约1700个问答对，均为需要多步推理的问题类型。验证集和测试集各分别约为1900和2600个问答对，同样均为需要多步推理的问题类型。第一阶段多步推理数据仅提供民事的一部分数据，规模较小，选手可充分利用CAIL2019的数据进行训练。

发放的文件为``train.json``，为字典列表，字典包含字段为：

- ``_id``：案例的唯一标识符。
- ``context``：案例内容，抽取自裁判文书的事实描述部分。数据格式与HotpotQA数据格式一致，不过只包含一个篇章，篇章包括标题（第一句话）和切割后的句子列表。
- ``question``：针对案例提出的问题，每个案例只标注一个问题。
- ``answer``：问题的回答，包括片段、YES/NO、据答几种类型，对于拒答类，答案应该是"unknown"。
- ``supporting_facts``：回答问题的依据，是个列表，每个元素包括标题（第一句话）和句子编号（从0开始）。

选手需根据案例描述和问题，给出答案及答案依据，最终会综合两部分的效果并作为评判依据，评价方法与HotpotQA一致。

## 提交的文件格式及组织形式

你可以在 ``model`` 中找到最简单的提交代码的格式。你需要将你所有的代码压缩为一个 ``zip`` 文件进行提交，该 ``zip`` 文件内部形式可以参看 ``model/submit_sample.zip``。该 ``zip`` 文件**内部顶层**必须包含``main.py``，为运行的入口程序，我们会在该目录下使用``python3 main.py``来运行你的程序。

## 代码的内容

对于你的代码，你需要从``../input/data.json``中读取数据进行预测，该数据格式与下发数据格式完全一致，隐去答案信息。选手需要将预测的结果输出到``../result/result.json``中，预测结果文件为一个json格式的文件，包含两个字段，分别为"answer"和"sp"，具体可以查看 ``evaluate/result.json``。

你可以利用 ``model`` 下的文件进行进一步参考。**请注意**，在加载模型的时候请尽量使用相对路径，我们会将提交的压缩包解压到``/model``路径下然后运行。

我们提供基于[RoBERTa-wwm-ext中文预训练模型](https://github.com/ymcui/Chinese-BERT-wwm)的基线模型（选手需自行训练），放置在``baseline``目录下，供选手参考。

## 评测脚本

我们在 ``evaluate`` 文件夹中提供了评分的代码，以供参考。

## 现有的系统环境

```
Package                          Version            
-------------------------------- -------------------
absl-py                          0.9.0              
anykeystore                      0.2                
apex                             0.1                
asn1crypto                       1.3.0              
astor                            0.8.1              
attrs                            19.3.0             
backcall                         0.1.0              
backports.functools-lru-cache    1.6.1              
backports.tempfile               1.0                
backports.weakref                1.0.post1          
beautifulsoup4                   4.9.0              
bert-serving-client              1.10.0             
bert-serving-server              1.10.0             
bleach                           3.1.5              
blis                             0.4.1              
boto                             2.49.0             
boto3                            1.13.3             
botocore                         1.16.3             
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
fastprogress                     0.2.3              
fasttext                         0.9.2              
filelock                         3.0.12             
Flask                            1.1.2              
future                           0.18.2             
gast                             0.2.2              
gensim                           3.8.3              
glob2                            0.7                
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
pandas                           1.0.4              
parso                            0.7.0              
PasteDeploy                      2.1.0              
pbkdf2                           1.3                
pbr                              3.1.1              
pexpect                          4.8.0              
pickleshare                      0.7.5              
Pillow                           7.0.0              
pip                              20.0.2             
pkginfo                          1.5.0.1            
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
Pygments                         2.6.1              
pyhanlp                          0.1.64             
pyltp                            0.2.1              
pyOpenSSL                        19.1.0             
pyparsing                        2.4.7              
pyramid                          1.10.4             
pyramid-mailer                   0.15.1             
pyrsistent                       0.16.0             
PySocks                          1.7.1              
python-dateutil                  2.8.1              
python-Levenshtein               0.12.0             
python3-openid                   3.1.0              
pytoml                           0.1.21             
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
tensorboard                      1.14.0             
tensorflow                       1.14.0             
tensorflow-estimator             2.1.0              
tensorflow-hub                   0.8.0              
termcolor                        1.1.0              
tflearn                          0.3.2              
Theano                           1.0.4              
thinc                            7.4.0              
thulac                           0.2.1              
tokenizers                       0.7.0              
torch                            1.4.0              
torchvision                      0.5.0              
tqdm                             4.46.0             
traitlets                        4.3.3              
transaction                      3.0.0              
transformers                     2.10.0             
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
```

另外评测使用的GPU型号是Tesla P40G24。

如果你有需要的环境，请联系比赛管理员进行安装。

## 法小飞
由哈工大讯飞联合实验室出品的“法小飞”智能法律咨询助手应用了对话型阅读理解技术，在为用户提供精准答案的同时提升了用户的对话交互体验。“法小飞”是一个服务公众和律师的法律咨询助手，旨在利用自然语言处理技术和法律专业知识，为用户提供快速优质的法律咨询及相关服务。“法小飞”通过学习大量的法律知识，对当事人提出的法律问题进行自动解答，并且能够针对刑事和民事案件进行深入的案情分析，拥有类案推送、法条推荐、判决预测和律师推荐的功能。

<div align=center><img width="400" height="400" src="https://s1.ax1x.com/2020/06/15/NCkdnf.jpg"/></div>

## 关注我们
欢迎关注哈工大讯飞联合实验室官方微信公众号，了解最新的技术动态。

<div align=center><img width="400" height="400" src="https://s1.ax1x.com/2020/06/17/NEvQu6.jpg"/></div>

## 问题反馈
如有问题，请在GitHub Issue中提交

