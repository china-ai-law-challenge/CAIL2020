import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_pretrained_bert import BertModel


class BertEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        """
        Initialize a list of configurations.

        Args:
            self: (todo): write your description
            config: (todo): write your description
            gpu_list: (list): write your description
            params: (dict): write your description
        """
        super(BertEncoder, self).__init__()

        self.bert = BertModel.from_pretrained(config.get("model", "bert_path"))

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        _, y = self.bert(x, output_all_encoded_layers=False)

        return y
