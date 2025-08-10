# @title losses/arcface.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFaceLoss(nn.Module):
    """
    Additive Angular Margin Loss for Deep Face Recognition.

    Reference:
    Jiankang Deng, Jia Guo, Niannan Xue, Stefanos Zafeiriou. ArcFace. CVPR 2019.

    Args:
        s (float): The scale parameter.
        m (float): The angular margin penalty in radians.
    """
    def __init__(self, s=30.0, m=0.50):
        super(ArcFaceLoss, self).__init__()
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, labels):
        """
        Args:
            cosine: The cosine similarity between features and weights, with shape (batch_size, num_classes).
            labels: Ground truth labels with shape (batch_size).
        """
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-7)
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, labels)