import torch
import torch.nn as nn
import torch.nn.functional as F

from model.encoder.LSTMEncoder import LSTMEncoder
from model.layer.Attention import Attention
from tools.accuracy_tool import single_label_top1_accuracy
from model.qa.util import generate_ans


class Model(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(Model, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.word_num = 0
        f = open(config.get("data", "word2id"), "r", encoding="utf8")
        for line in f:
            self.word_num += 1

        self.embedding = nn.Embedding(self.word_num, self.hidden_size)
        self.context_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.question_encoder = LSTMEncoder(config, gpu_list, *args, **params)
        self.attention = Attention(config, gpu_list, *args, **params)

        self.rank_module = nn.Linear(self.hidden_size * 2, 1)

        self.criterion = nn.CrossEntropyLoss()

        self.multi_module = nn.Linear(4, 16)
        self.accuracy_function = single_label_top1_accuracy

    def init_multi_gpu(self, device, config, *args, **params):
        pass

    def forward(self, data, config, gpu_list, acc_result, mode):
        context = data["context"]
        question = data["question"]

        batch = question.size()[0]
        option = question.size()[1]

        context = context.view(batch * option, -1)
        question = question.view(batch * option, -1)
        context = self.embedding(context)
        question = self.embedding(question)

        _, context = self.context_encoder(context)
        _, question = self.question_encoder(question)

        c, q, a = self.attention(context, question)

        y = torch.cat([torch.max(c, dim=1)[0], torch.max(q, dim=1)[0]], dim=1)

        y = y.view(batch * option, -1)
        y = self.rank_module(y)

        y = y.view(batch, option)

        y = self.multi_module(y)

        if mode != "test":
            label = data["label"]
            loss = self.criterion(y, label)
            acc_result = self.accuracy_function(y, label, config, acc_result)
            return {"loss": loss, "acc_result": acc_result}

        return {"output": generate_ans(data["id"], y)}
