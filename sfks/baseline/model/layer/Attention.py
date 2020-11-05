import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, config, gpu_list, *args, **params):
        """
        Initialize the network.

        Args:
            self: (todo): write your description
            config: (todo): write your description
            gpu_list: (list): write your description
            params: (dict): write your description
        """
        super(Attention, self).__init__()

        self.hidden_size = config.getint("model", "hidden_size")
        self.fc = nn.Linear(self.hidden_size, self.hidden_size)

    def forward(self, x, y):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            y: (todo): write your description
        """
        x_ = x  # self.fc(x)
        y_ = torch.transpose(y, 1, 2)
        a_ = torch.bmm(x_, y_)

        x_atten = torch.softmax(a_, dim=2)
        x_atten = torch.bmm(x_atten, y)

        y_atten = torch.softmax(a_, dim=1)
        y_atten = torch.bmm(torch.transpose(y_atten, 2, 1), x)

        return x_atten, y_atten, a_
