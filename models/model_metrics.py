
import torch

#Idk what the threshold should actually be so I defaulted to 0.5
# Also according to chatgpt i should have this epsilon value but idk what it does honestly my guess is it is to account for floating point precision stuff
def batch_metrics(probs, masks, threshold=0.5, eps=1e-8):
    preds = (probs > threshold).float()

    tp = (preds * masks).sum(dim=(1, 2, 3))
    fp = (preds * (1 - masks)).sum(dim=(1, 2, 3))
    fn = ((1 - preds) * masks).sum(dim=(1, 2, 3))

    dice = (2 * tp) / (2 * tp + fp + fn + eps)
    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    beta2 = 4.0
    f2 = (1 + beta2) * precision * recall / (beta2 * precision + recall + eps)

    mae = torch.abs(probs - masks).mean(dim=(1, 2, 3))

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f2": f2,
        "mae": mae,
    }


class MetricTracker:

    def __init__(self):
        self.values = {k: [] for k in ("dice", "iou", "precision", "recall", "f2", "mae")}

    def update(self, probs, masks, threshold=0.5):
        batch = batch_metrics(probs, masks, threshold)
        for k, v in batch.items():
            self.values[k].append(v.detach().cpu())

    def compute(self):
        return {k: torch.cat(v).mean().item() for k, v in self.values.items()}

    def report(self, prefix=""):
        m = self.compute()
        print(
            f"{prefix}Dice {m['dice']:.4f} | IoU {m['iou']:.4f} | "
            f"Prec {m['precision']:.4f} | Rec {m['recall']:.4f} | "
            f"F2 {m['f2']:.4f} | MAE {m['mae']:.4f}"
        )
        return m
