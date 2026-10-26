import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginAwareCELoss(nn.Module):
    """
    Margin-Aware Cross Entropy Loss

    Clean      -> small weight
    Ambiguous  -> large weight
    Noisy      -> ~0 weight
    """

    def __init__(self, reduction: str = "mean", eps: float = 1e-8):
        super().__init__()
        self.reduction = reduction
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        probs = F.softmax(logits, dim=1)  # (N, C)

        p_gt = probs.gather(1, targets.unsqueeze(1))  # (N,1)


        mask = torch.ones_like(probs).scatter_(1, targets.unsqueeze(1), 0.0)
        probs_non_gt = probs * mask
        p_nn, _ = probs_non_gt.max(dim=1, keepdim=True)  # (N,1)


        margin = (p_gt - p_nn) / (p_gt + p_nn + self.eps)
        weights = 1.0 - margin.pow(2)


        ce_loss = F.cross_entropy(logits, targets, reduction="none").unsqueeze(1)

        loss = (1.0 + weights) * ce_loss

        loss = loss.squeeze(1)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class BalancedMarginAwareCELoss(nn.Module):
    """
    Margin-Aware Cross Entropy with stop-gradient weights, logit adjustment and label smoothing.

    weight = sech^2(excess / temperature), from the detached logit margin z_y - max_{j != y} z_j
             (MarginAwareCELoss's margin equals tanh(logit margin / 2), so its weight is sech^2(logit margin / 2))
    excess = margin if margin >= 0, else max(0, -margin - delta)   (near-miss errors keep the full weight)
    loss   = (1 + lam * weight) * CE_smoothed(logits + tau * log(class prior), y)

    tau=0, smoothing=0, temperature=2, delta=0, lam=1 gives the same values as MarginAwareCELoss.
    smoothing=2/15 equals CE + 2 * LabelSmoothingCrossEntropy(0.2), divided by 3.
    The prior shift is for training only: predict with the raw logits.
    Call set_epoch(epoch): lam is 0 for warmup_epochs epochs, then ramps up to 1 over ramp_epochs.
    """

    def __init__(self, class_counts, tau=0.25, smoothing=2 / 15, temperature=2.0, delta=1.0,
                 warmup_epochs=10, ramp_epochs=5):
        super().__init__()
        prior = torch.as_tensor(class_counts, dtype=torch.float).clamp(min=1)
        self.register_buffer("log_prior", (prior / prior.sum()).log())
        self.tau = tau
        self.smoothing = smoothing
        self.temperature = temperature
        self.delta = delta
        self.warmup_epochs = warmup_epochs
        self.ramp_epochs = ramp_epochs
        self.lam = 1.0

    def set_epoch(self, epoch: int):
        done = epoch - self.warmup_epochs + 1
        if self.ramp_epochs > 0:
            self.lam = min(1.0, max(0.0, done / self.ramp_epochs))
        else:
            self.lam = 1.0 if done > 0 else 0.0

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):

        z = logits + self.tau * self.log_prior

        with torch.no_grad():
            z_gt = z.gather(1, targets.unsqueeze(1)).squeeze(1)
            z_other = z.scatter(1, targets.unsqueeze(1), float("-inf"))
            margin = z_gt - z_other.max(dim=1).values
            excess = torch.where(margin >= 0, margin, (-margin - self.delta).clamp(min=0))
            weights = torch.cosh(excess / self.temperature).pow(-2)

        ce_loss = F.cross_entropy(z, targets, reduction="none", label_smoothing=self.smoothing)

        return ((1.0 + self.lam * weights) * ce_loss).mean()


class LabelSmoothingCrossEntropy(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x, target):
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

