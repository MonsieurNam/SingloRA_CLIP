# @title losses/max_entropy.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaximumEntropyLoss(nn.Module):
    """
    Maximum Entropy Regularization.

    This loss encourages the model to produce a more uniform probability distribution
    over the classes, which can act as a regularizer to prevent overconfidence.

    The entropy is calculated as H(p) = - sum(p_i * log(p_i)).
    We want to *maximize* this entropy, which is equivalent to *minimizing* its negative.

    Args:
        use_softmax (bool): Whether to apply softmax to the input logits first.
    """
    def __init__(self, use_softmax=True):
        super(MaximumEntropyLoss, self).__init__()
        self.use_softmax = use_softmax

    def forward(self, logits):
        """
        Args:
            logits: The model's output logits, with shape (batch_size, num_classes).
        """
        if self.use_softmax:
            probs = F.softmax(logits, dim=1)
        else:
            probs = logits
        probs = torch.clamp(probs, min=1e-7)
        entropy_per_sample = -torch.sum(probs * torch.log(probs), dim=1)
        return torch.mean(entropy_per_sample)