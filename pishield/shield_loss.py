"""Top-level entry point for building Shield Losses.

A Shield Loss is an additional loss term that *encourages* (but, unlike a Shield
Layer, does not guarantee) requirement satisfaction at training time, using
t-norms. This module exposes :func:`build_shield_loss`.
"""

from pishield.propositional_requirements.shield_loss import ShieldLoss

def build_shield_loss(num_variables: int,
                       requirements_filepath: str,
                       tnorm_choice: str = 'godel',
                       requirements_type='propositional'):
    """Build a Shield Loss from the given requirements.

    Constructs a loss term that penalises violations of the requirements during
    training, using the chosen t-norm to measure satisfaction.

    Args:
        num_variables: Total number of variables (e.g. labels or features,
            depending on the task), matching the dimension of the tensors that
            are scored by the loss.
        requirements_filepath: Path to a ``.txt`` file containing the requirements.
        tnorm_choice: The t-norm used to compute requirement satisfaction. One of
            ``'product'``, ``'godel'``, or ``'lukasiewicz'``. Defaults to ``'godel'``.
        requirements_type: The requirement type. Only ``'propositional'`` is
            currently supported.

    Returns:
        A ``ShieldLoss`` instance that computes the requirement-satisfaction loss.

    Raises:
        Exception: If ``requirements_type`` is not ``'propositional'``.

    Example:
        >>> loss_fn = build_shield_loss(
        ...     num_variables=5,
        ...     requirements_filepath='requirements.txt',
        ...     tnorm_choice='product',
        ... )
        >>> penalty = loss_fn(model_output)
    """

    if requirements_type == 'propositional':
        return ShieldLoss(num_variables, requirements_filepath, tnorm_choice)
    else:
        raise Exception('Unknown requirements type!')
