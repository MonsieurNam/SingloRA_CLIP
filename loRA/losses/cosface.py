# @title losses/cosface.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class CosFaceLoss(nn.Module):
    """
    Large Margin Cosine Loss for Deep Face Recognition.

    Reference:
    Hao Wang, Yitong Wang, Zheng Zhou, Xingji Ji, Dihong Gong, Jingchao Zhou, Zhifeng Li, and Wei Liu. CosFace. CVPR 2018.

    Args:
        s (float): The scale parameter.
        m (float): The cosine margin penalty.
    """
    def __init__(self, s=30.0, m=0.35):
        super(CosFaceLoss, self).__init__()
        self.s = s
        self.m = m

    def forward(self, cosine, labels):
        """
        Args:
            cosine: The cosine similarity between features and weights, with shape (batch_size, num_classes).
            labels: Ground truth labels with shape (batch_size).
        """
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        phi = cosine - self.m
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return F.cross_entropy(output, labels)