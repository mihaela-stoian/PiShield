import torch


class DetectionThreshold:
    def __init__(self, threshold):
        self.threshold = threshold

    def cut(self, preds, mask):
        return preds[mask, 1:], lambda updated: self.uncut(preds, mask, updated)

    def uncut(self, init, mask, preds):
        preds = torch.cat((init[mask, 0].reshape(-1, 1), preds), dim=1)
        index = torch.tensor(list(range(init.shape[0])), device=mask.device)
        index = index[mask]

        # init = torch.cat((init[:, 0].reshape(-1, 1), torch.zeros_like(init[:, 1:])), dim=1)
        return init.index_copy(0, index, preds)

    def cutter(self, preds):
        mask = preds[:, 0] > self.threshold
        return lambda preds: self.cut(preds, mask)
