import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class MultiLabelSoftmaxLoss(nn.Module):
    def __init__(self, config):
        """
        Initialize the gradient

        Args:
            self: (todo): write your description
            config: (todo): write your description
        """
        super(MultiLabelSoftmaxLoss, self).__init__()
        self.task_num = config.getint("model", "output_dim")
        self.criterion = []
        for a in range(0, self.task_num):
            try:
                ratio = config.getfloat("train", "loss_weight_%d" % a)
                self.criterion.append(
                    nn.CrossEntropyLoss(weight=torch.from_numpy(np.array([1.0, ratio], dtype=np.float32)).cuda()))
                # print_info("Task %d with weight %.3lf" % (task, ratio))
            except Exception as e:
                self.criterion.append(nn.CrossEntropyLoss())

    def forward(self, outputs, labels):
        """
        Forward computation on all outputs

        Args:
            self: (todo): write your description
            outputs: (todo): write your description
            labels: (todo): write your description
        """
        loss = 0
        for a in range(0, len(outputs[0])):
            o = outputs[:, a, :].view(outputs.size()[0], -1)
            loss += self.criterion[a](o, labels[:, a])

        return loss


def multi_label_cross_entropy_loss(outputs, labels):
    """
    Calculate the loss.

    Args:
        outputs: (todo): write your description
        labels: (todo): write your description
    """
    labels = labels.float()
    temp = outputs
    res = - labels * torch.log(temp) - (1 - labels) * torch.log(1 - temp)
    res = torch.mean(torch.sum(res, dim=1))

    return res


def cross_entropy_loss(outputs, labels):
    """
    Calculate the loss.

    Args:
        outputs: (todo): write your description
        labels: (todo): write your description
    """
    criterion = nn.CrossEntropyLoss()
    return criterion(outputs, labels)


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        """
        Initialize the gamma.

        Args:
            self: (todo): write your description
            gamma: (float): write your description
            alpha: (float): write your description
            size_average: (int): write your description
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average

    def forward(self, input, target):
        """
        Forward computation.

        Args:
            self: (todo): write your description
            input: (todo): write your description
            target: (todo): write your description
        """
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()
