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
        # Apply softmax to convert logits to probabilities
        if self.use_softmax:
            probs = F.softmax(logits, dim=1)
        else:
            # Assumes input is already a probability distribution
            probs = logits

        # Clamp probabilities to avoid log(0) = -inf
        probs = torch.clamp(probs, min=1e-7)

        # Calculate entropy for each sample in the batch
        # H(p) = - sum(p_i * log(p_i))
        entropy_per_sample = -torch.sum(probs * torch.log(probs), dim=1)

        # We want to MAXIMIZE entropy, so we MINIMIZE its negative.
        # However, it's more common to see it as subtracting the entropy from the main loss.
        # Here, we will return the POSITIVE entropy, and the main training loop will
        # subtract it from the CE loss, scaled by a weight.
        # L_total = L_CE - weight * H(p)
        # So we just need to return the mean entropy.

        return torch.mean(entropy_per_sample)