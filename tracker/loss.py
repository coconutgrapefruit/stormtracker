import torch
import torch.nn as nn
from tracker.train import calc_loss


class EuclideanLoss(nn.Module):
    def __init__(self, area=(70, 120, 0, 220)):
        super(EuclideanLoss, self).__init__()
        self.area = area

    def forward(self, x, y):
        return calc_loss(x, y, y.device, self.area)
