import torch


class SegmentationMetrics:
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.tp = 0.0
        self.fp = 0.0
        self.fn = 0.0
        self.tn = 0.0

    @torch.no_grad()
    def update(self, logits: torch.Tensor, target: torch.Tensor):
        pred = (torch.sigmoid(logits) >= self.threshold).float()
        target = (target >= 0.5).float()
        self.tp += (pred * target).sum().item()
        self.fp += (pred * (1 - target)).sum().item()
        self.fn += ((1 - pred) * target).sum().item()
        self.tn += ((1 - pred) * (1 - target)).sum().item()

    def compute(self):
        eps = 1e-7
        precision = self.tp / (self.tp + self.fp + eps)
        recall = self.tp / (self.tp + self.fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = self.tp / (self.tp + self.fp + self.fn + eps)
        dice = 2 * self.tp / (2 * self.tp + self.fp + self.fn + eps)
        accuracy = (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn + eps)
        return {
            "iou": iou,
            "dice": dice,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }
