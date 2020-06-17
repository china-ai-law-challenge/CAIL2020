import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTMEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        super(LSTMEncoder, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.bi = config.getboolean("model", "bi_direction")
        self.output_size = self.hidden_size
        self.num_layers = config.getint("model", "num_layers")
        if self.bi:
            self.output_size = self.output_size // 2

        self.lstm = nn.LSTM(input_size=self.hidden_size, hidden_size=self.output_size,
                            num_layers=self.num_layers, batch_first=True, bidirectional=self.bi)

    def forward(self, x):
        batch_size = x.size()[0]
        seq_len = x.size()[1]
        # print(x.size())
        # print(batch_size, self.num_layers + int(self.bi) * self.num_layers, self.output_size)
        hidden = (
            torch.autograd.Variable(
                torch.zeros(self.num_layers + int(self.bi) * self.num_layers, batch_size, self.output_size)).cuda(),
            torch.autograd.Variable(
                torch.zeros(self.num_layers + int(self.bi) * self.num_layers, batch_size, self.output_size)).cuda())

        h_, c = self.lstm(x, hidden)

        h = torch.max(h_, dim=1)[0]

        return h, h_
