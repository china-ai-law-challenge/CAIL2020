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

我们在 ``evaluate`` 文件夹中提供了评分的代码，以供参考。评测代码正在更新中....

## 现有的系统环境

```
Package             Version               
------------------- ----------------------
absl-py==0.9.0
anykeystore==0.2
apex==0.1
asn1crypto==1.3.0
astor==0.8.1
attrs==19.3.0
backcall==0.1.0
backports.functools-lru-cache==1.6.1
backports.tempfile==1.0
backports.weakref==1.0.post1
beautifulsoup4==4.9.0
bert-serving-client==1.10.0
bert-serving-server==1.10.0
bleach==3.1.5
blis==0.4.1
boto==2.49.0
boto3==1.13.3
botocore==1.16.3
Bottleneck==1.3.2
bz2file==0.98
cachetools==4.1.0
catalogue==1.0.0
certifi==2020.4.5.1
cffi==1.14.0
chardet==3.0.4
click==7.1.2
conda==4.8.3
conda-build==3.4.2
conda-package-handling==1.6.0
conda-verify==3.4.2
cryptacular==1.5.5
cryptography==2.8
cycler==0.10.0
cymem==2.0.3
Cython==0.29.17
dataclasses==0.7
decorator==4.4.2
defusedxml==0.6.0
docutils==0.15.2
fastai==1.0.61
fastprogress==0.2.3
fasttext==0.9.2
filelock==3.0.12
Flask==1.1.2
future==0.18.2
gast==0.2.2
gensim==3.8.3
glob2==0.7
google-auth==1.14.1
google-auth-oauthlib==0.4.1
google-pasta==0.2.0
GPUtil==1.4.0
grpcio==1.28.1
h5py==2.10.0
html5lib==1.0.1
hupper==1.10.2
idna==2.9
importlib-metadata==1.6.0
ipython==7.14.0
ipython-genutils==0.2.0
itsdangerous==1.1.0
jedi==0.17.0
jeepney==0.4.3
jieba==0.42.1
Jinja2==2.11.2
jmespath==0.9.5
joblib==0.14.1
JPype1==0.7.0
jsonschema==3.2.0
Keras==2.3.1
Keras-Applications==1.0.8
keras-bert==0.81.0
keras-embed-sim==0.7.0
keras-layer-normalization==0.14.0
keras-multi-head==0.22.0
keras-pos-embd==0.11.0
keras-position-wise-feed-forward==0.6.0
Keras-Preprocessing==1.1.0
keras-self-attention==0.41.0
keras-transformer==0.33.0
keyring==21.2.1
keyrings.alt==3.4.0
kiwisolver==1.2.0
lda==1.1.0
lightgbm==2.3.1
Mako==1.1.2
Markdown==3.2.1
MarkupSafe==1.1.1
matplotlib==3.2.1
mkl-fft==1.0.15
mkl-random==1.1.0
mkl-service==2.3.0
mock==4.0.2
murmurhash==1.0.2
ninja==1.9.0.post1
nltk==3.5
numexpr==2.7.1
numpy==1.18.1
nvidia-ml-py3==7.352.0
oauthlib==3.1.0
olefile==0.46
opt-einsum==3.2.1
packaging==20.3
pandas==1.0.3
parso==0.7.0
PasteDeploy==2.1.0
pbkdf2==1.3
pbr==3.1.1
pexpect==4.8.0
pickleshare==0.7.5
Pillow==7.0.0
pkginfo==1.5.0.1
plac==1.1.3
plaster==1.0
plaster-pastedeploy==0.7
preshed==3.0.2
prompt-toolkit==3.0.5
protobuf==3.11.3
ptyprocess==0.6.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pybind11==2.5.0
pycosat==0.6.3
pycparser==2.20
pycrypto==2.6.1
pycurl==7.43.0.5
Pygments==2.6.1
pyhanlp==0.1.64
pyOpenSSL==19.1.0
pyparsing==2.4.7
pyramid==1.10.4
pyramid-mailer==0.15.1
pyrsistent==0.16.0
PySocks==1.7.1
python-dateutil==2.8.1
python3-openid==3.1.0
pytorch-pretrained-bert==0.6.2
pytorch-transformers==1.2.0
pytz==2020.1
pyxdg==0.26
PyYAML==5.3.1
pyzmq==19.0.1
regex==2020.4.4
repoze.sendmail==4.4.1
requests==2.23.0
requests-oauthlib==1.3.0
rsa==4.0
ruamel-yaml==0.15.87
s3transfer==0.3.3
sacremoses==0.0.43
scikit-learn==0.22.2.post1
scikit-multilearn==0.2.0
scipy==1.4.1
SecretStorage==3.1.2
sentencepiece==0.1.86
simplegeneric==0.8.1
six==1.14.0
sklearn==0.0
smart-open==2.0.0
soupsieve==2.0
spacy==2.2.4
SQLAlchemy==1.3.16
srsly==1.0.2
taboo==0.8.8
tensorboard==2.1.1
tensorflow==2.1.0
tensorflow-estimator==2.1.0
tensorflow-hub==0.8.0
termcolor==1.1.0
tflearn==0.3.2
Theano==1.0.4
thinc==7.4.0
thulac==0.2.1
tokenizers==0.7.0
torch==1.4.0
torchvision==0.5.0
tqdm==4.46.0
traitlets==4.3.3
transaction==3.0.0
transformers==2.10.0
translationstring==1.3
typing==3.7.4.1
urllib3==1.25.8
velruse==1.1.1
venusian==3.0.0
wasabi==0.6.0
wcwidth==0.1.9
webencodings==0.5.1
WebOb==1.8.6
Werkzeug==1.0.1
wrapt==1.12.1
WTForms==2.3.1
wtforms-recaptcha==0.3.2
xgboost==1.0.2
zipp==3.1.0
zope.deprecation==4.4.0
zope.interface==5.1.0
zope.sqlalchemy==1.3

```

等待补全中

如果你有需要的环境，请联系比赛管理员进行安装。
