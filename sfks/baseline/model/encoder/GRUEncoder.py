import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUEncoder(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            config: (todo): write your description
            gpu_list: (list): write your description
            params: (dict): write your description
        """
        super(GRUEncoder, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.bi = config.getboolean("model", "bi_direction")
        self.output_size = self.hidden_size
        self.num_layers = config.getint("model", "num_layers")
        if self.bi:
            self.output_size = self.output_size // 2

        self.gru = nn.GRU(input_size=self.hidden_size, hidden_size=self.output_size, num_layers=self.num_layers,
                          batch_first=True, bidirectional=self.bi)

    def forward(self, x):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        h_, c = self.gru(x)

        h = torch.max(h_, dim=1)[0]

        return h, h_
