# -*- coding:utf-8 -*-


import torch
import torch.nn as nn
import torch.nn.functional as F


class PolyLoss(nn.Module):
    def __init__(self, weight_loss, DEVICE, epsilon=1.0):
        super(PolyLoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_loss, reduction='none')
        self.epsilon = epsilon
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        one_hot = torch.zeros((16, 2), device=self.DEVICE).scatter_(
            1, torch.unsqueeze(labels, dim=-1), 1)
        pt = torch.sum(one_hot * F.softmax(predicted, dim=1), dim=-1)
        ce = self.CELoss(predicted, labels)
        poly1 = ce + self.epsilon * (1-pt)
        return torch.mean(poly1)


class CELoss(nn.Module):
    def __init__(self, weight_CE, DEVICE):
        super(CELoss, self).__init__()
        self.CELoss = nn.CrossEntropyLoss(weight=weight_CE)
        self.DEVICE = DEVICE

    def forward(self, predicted, labels):
        return self.CELoss(predicted, labels)
