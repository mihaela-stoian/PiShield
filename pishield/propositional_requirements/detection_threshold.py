"""Detection-threshold based slicing of predictions.

A detection variable (at column 0) gates whether an example is processed: only rows
whose detection probability exceeds a threshold are kept (``cut``) before the
requirements are applied, and the corrected rows are then scattered back into the
original tensor (``uncut``).
"""

import torch


class DetectionThreshold:
    """Restrict prediction rows to those passing a detection threshold.

    Attributes:
        threshold: The minimum value of the detection variable (column 0) for a row
            to be kept.
    """

    def __init__(self, threshold):
        """Store the detection threshold.

        Args:
            threshold: The minimum detection probability for a row to be kept.
        """
        self.threshold = threshold

    def cut(self, preds, mask):
        """Drop the detection column and keep only masked rows.

        Args:
            preds: The full prediction tensor, shape (batch, num_classes).
            mask: A boolean row mask selecting which examples to keep.

        Returns:
            A tuple of the sliced predictions (kept rows, columns from index 1
            onward) and a callable that scatters updated values back via :meth:`uncut`.
        """
        return preds[mask, 1:], lambda updated: self.uncut(preds, mask, updated)

    def uncut(self, init, mask, preds):
        """Scatter corrected rows back into the original tensor.

        Re-attaches the detection column and writes the corrected rows back into the
        positions selected by ``mask``, leaving non-selected rows untouched.

        Args:
            init: The original full prediction tensor before cutting.
            mask: The boolean row mask used by :meth:`cut`.
            preds: The corrected predictions for the kept rows (without column 0).

        Returns:
            A tensor the shape of ``init`` with corrected rows written back in place.
        """
        preds = torch.cat((init[mask, 0].reshape(-1, 1), preds), dim=1)
        index = torch.tensor(list(range(init.shape[0])), device=mask.device)
        index = index[mask]

        # init = torch.cat((init[:, 0].reshape(-1, 1), torch.zeros_like(init[:, 1:])), dim=1)
        return init.index_copy(0, index, preds)

    def cutter(self, preds):
        """Build a cut function for predictions using the detection threshold.

        Args:
            preds: A prediction tensor whose column 0 is the detection variable.

        Returns:
            A callable mapping a prediction tensor to the result of :meth:`cut` using
            a row mask derived from this tensor's detection column.
        """
        mask = preds[:, 0] > self.threshold
        return lambda preds: self.cut(preds, mask)
