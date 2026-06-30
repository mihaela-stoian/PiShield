"""Training, evaluation and ordering utilities for propositional requirements.

Helper routines used in examples and experiments: a standard PyTorch training loop and
test loop that run predictions through a Shield Layer, a function for visualising the
class predictions of a 2D model, and a helper for resolving the variable ordering /
centrality used to stratify the requirements.
"""

import torch
import numpy as np


def train(dataloader, model, clayer, loss_fn, optimizer, device, ratio=1.):
    """Run one training epoch with predictions corrected by the Shield Layer.

    For each batch, the model's predictions are passed through ``clayer`` (with the
    ground-truth labels as goal and a slicer that gradually enables the requirements),
    then the loss is computed on the sliced atoms and backpropagated.

    Args:
        dataloader: A PyTorch DataLoader yielding ``(X, y)`` batches.
        model: The prediction model.
        clayer: The Shield Layer correcting the predictions.
        loss_fn: The loss function applied to corrected predictions and labels.
        optimizer: The optimizer updating the model parameters.
        device: The torch device to run on.
        ratio: The fraction of strata to enable via the layer's slicer (1.0 = all).
    """
    size = len(dataloader.dataset)
    model, clayer = model.to(device), clayer.to(device)
    slicer = clayer.slicer(ratio)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        constrained = clayer(pred, goal=y, slicer=slicer)

        constrained, y = slicer.slice_atoms(constrained), slicer.slice_atoms(y)
        loss = loss_fn(constrained, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f} [{current:>5d}/{size:>5d}]")


def test(dataloader, model, clayer, loss_fn, device):
    """Evaluate the model with Shield-Layer-corrected predictions.

    Runs the model over the dataloader (no goal), corrects the predictions with
    ``clayer``, and reports the average loss and per-class accuracy.

    Args:
        dataloader: A PyTorch DataLoader yielding ``(X, y)`` batches.
        model: The prediction model.
        clayer: The Shield Layer correcting the predictions.
        loss_fn: The loss function.
        device: The torch device to run on.

    Returns:
        A tuple ``(test_loss, correct)`` of the average loss and the list of per-class
        accuracy percentages.
    """
    size = len(dataloader.dataset)
    model, clayer = model.to(device), clayer.to(device)
    model.eval()

    test_loss = 0.
    correct = 0.

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X)
            pred = clayer(pred)
            test_loss += loss_fn(pred, y).item()
            correct += (torch.where(pred > 0.5, 1., 0.) == y).sum(dim=0)

    test_loss /= size
    correct /= size

    correct = [100 * rate for rate in correct]
    accuracy = ", ".join([f"{rate:>0.1f}%" for rate in correct])
    print(f"Test Error: \n Accuracy: {accuracy}")
    print(f" Avg loss: {test_loss:>8f} \n")
    return test_loss, correct


def draw_classes(model, draw=None, path=None, device='cpu', show=False):
    """Plot each class score of a model over the unit square.

    Evaluates the model on a dense grid of points in ``[0, 1)^2`` and renders one
    filled-contour subplot per output class.

    Args:
        model: A model mapping 2D points to per-class scores.
        draw: Optional callable ``draw(ax, i)`` to overlay extra artwork on subplot i.
        path: Optional file path to save the figure to.
        device: The torch device to run the model on.
        show: If True, display the figure interactively.

    Returns:
        The matplotlib Figure that was created.
    """
    import matplotlib.pyplot as plt  # imported lazily: plotting is optional and pulls in a heavy dependency
    dots = np.arange(0., 1., 0.001, dtype="float32")
    grid = torch.tensor([(x, y) for y in dots for x in dots]).to(device)
    model = model.to(device)
    preds = model(grid).detach()

    classes = preds.shape[1]
    fig, ax = plt.subplots(1, classes, figsize=(20, 20 * classes))
    for i, ax in enumerate(ax):
        image = preds[:, i].view((len(dots), len(dots))).to('cpu')
        # ax.imshow(
        #     image, 
        #     cmap='hot', 
        #     interpolation='nearest', 
        #     origin='lower', 
        #     extent=(0., 1., 0., 1.),
        #     vmin=0.,
        #     vmax=1.
        # )
        ax.contourf(
            dots,
            dots,
            image,
            cmap='hot',
            origin='lower',
            extent=(0., 1., 0., 1.),
            vmin=0.1,
            vmax=1.
        )
        if draw != None: draw(ax, i)

    if show:
        plt.show()

    if not path is None:
        plt.savefig(path)
        plt.close()

    return fig


def get_order_and_centrality(ordering_choice: str, custom_ordering: str):
    """Resolve the centrality/ordering argument used to stratify the requirements.

    If a custom ordering is supplied and the choice is a custom/given one, the ordering
    is parsed into an explicit array of atom indices (reversed when the choice contains
    ``'rev'``); otherwise the named centrality choice is returned as-is.

    Args:
        ordering_choice: The ordering choice name (e.g. a centrality measure, or one
            containing ``'custom'``/``'given'``, optionally with ``'rev'``).
        custom_ordering: An optional comma-separated string of atom indices.

    Returns:
        Either the ordering choice name (str) or an array of atom indices giving an
        explicit order.
    """
    if custom_ordering is None:
        return ordering_choice
    if 'custom' in ordering_choice or 'given' in ordering_choice:
        order = custom_ordering.split(',')
        centrality = np.array([int(nr) for nr in order])
        if 'rev' in ordering_choice:
            centrality = centrality[::-1]
    else:
        centrality = ordering_choice
    return centrality
