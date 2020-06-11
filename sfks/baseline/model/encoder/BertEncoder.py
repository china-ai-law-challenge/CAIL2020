import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(BertEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

    def forward(self, x):
        _, y = self.bert(x, output_all_encoded_layers=False)

        return y
