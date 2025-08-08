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
        # A small value to prevent acos from returning NaN
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, cosine, labels):
        """
        Args:
            cosine: The cosine similarity between features and weights, with shape (batch_size, num_classes).
            labels: Ground truth labels with shape (batch_size).
        """
        # 1. Calculate the angle (theta) from cosine
        # Add a small epsilon for numerical stability
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2) + 1e-7)

        # 2. Calculate cos(theta + m) using the angle addition formula
        # phi = cos(theta + m) = cos(theta)*cos(m) - sin(theta)*sin(m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # 3. Ensure that for theta > (pi - m), phi is still monotonically decreasing
        # This is a trick from the original paper to make optimization stable.
        # It's equivalent to phi = cos(theta) - m when theta + m > pi.
        # We use a hard-coded version for simplicity.
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 4. Create a one-hot mask for the ground truth classes
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # 5. Assemble the final output logits
        # Where the label is correct, use the modified logit (phi)
        # Where the label is incorrect, use the original logit (cosine)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

        # 6. Scale the output
        output *= self.s

        # 7. Calculate the final cross-entropy loss
        return F.cross_entropy(output, labels)