"""Top-level entry point for building Shield Layers.

A Shield Layer is a differentiable layer that corrects a model's outputs so
that they are *guaranteed* to satisfy a given set of requirements (constraints),
regardless of the input. This module exposes :func:`build_shield_layer`, which
dispatches to the appropriate backend (linear, QFLRA, or propositional) based on
the requirements, and :func:`detect_requirements_type`, which infers the
requirement type from a requirements file.
"""

from typing import List

from pishield.linear_requirements.shield_layer import ShieldLayer as LinearConstraintLayer
from pishield.qflra_requirements.shield_layer import ShieldLayer as QFLRAConstraintLayer
from pishield.propositional_requirements.shield_layer import ShieldLayer as PropositionalConstraintLayer
from pishield.hierarchical_requirements.shield_layer import ShieldLayer as HierarchicalConstraintLayer


def build_shield_layer(num_variables: int,
                       requirements_filepath: str,
                       ordering_choice: str = 'given',
                       custom_ordering: List = None,
                       requirements_type='auto',
                       arff_hierarchy_style: str = 'auto'):
    """Build a Shield Layer from the given requirements.

    Selects and constructs the appropriate Shield Layer backend (linear, QFLRA,
    or propositional) for the supplied requirements. The returned layer is
    differentiable and can be used at both inference and training time to correct
    a model's outputs so that they satisfy the requirements.

    Args:
        num_variables: Total number of variables (e.g. labels or features,
            depending on the task), matching the dimension of the tensors that
            are to be corrected by the layer.
        requirements_filepath: Path to a ``.txt`` file containing the requirements.
        ordering_choice: How to order the variables when applying corrections.
            One of ``'given'``, ``'random'``, or a custom ordering implemented by
            the user. If ``'given'``, the ordering is read from
            ``requirements_filepath`` when available, otherwise the ascending
            order of the variables is used. If ``'random'``, a random ordering of
            the variables is used.
        custom_ordering: An explicit ordering of the variables (only used by the
            propositional backend). Defaults to None.
        requirements_type: One of ``'auto'``, ``'linear'``, ``'propositional'``,
            ``'qflra'``, or ``'hierarchical'``. If ``'auto'``, the appropriate
            backend is detected from the contents (or extension) of
            ``requirements_filepath`` via :func:`detect_requirements_type`.
        arff_hierarchy_style: For hierarchical requirements supplied as an
            ``.arff`` file, the hierarchy style: ``'auto'`` (the default) detects
            it from the file, or force ``'paths'`` (FUN-style) or ``'edges'``
            (GO-style). Ignored by the other backends.

    Returns:
        A Shield Layer instance (``LinearConstraintLayer``, ``QFLRAConstraintLayer``,
        ``PropositionalConstraintLayer``, or ``HierarchicalConstraintLayer``) that
        corrects model outputs to satisfy the requirements.

    Raises:
        Exception: If ``requirements_type`` is not one of the recognised values.

    Example:
        >>> layer = build_shield_layer(
        ...     num_variables=5,
        ...     requirements_filepath='requirements.txt',
        ... )
        >>> corrected = layer(model_output)  # corrected satisfies the requirements
    """

    if requirements_type == 'linear':
        return LinearConstraintLayer(num_variables, requirements_filepath, ordering_choice)
    elif requirements_type == 'qflra':
        return QFLRAConstraintLayer(num_variables, requirements_filepath, ordering_choice)
    elif requirements_type == 'propositional':
        return PropositionalConstraintLayer(num_variables, requirements_filepath, ordering_choice, custom_ordering=custom_ordering)
    elif requirements_type == 'hierarchical':
        return HierarchicalConstraintLayer(num_variables, requirements_filepath, ordering_choice,
                                           arff_hierarchy_style=arff_hierarchy_style)
    elif requirements_type == 'auto':
        detected_requirements_type = detect_requirements_type(requirements_filepath)
        return build_shield_layer(num_variables, requirements_filepath, ordering_choice, custom_ordering=custom_ordering,
                                  requirements_type=detected_requirements_type, arff_hierarchy_style=arff_hierarchy_style)
    else:
        raise Exception('Unknown requirements type!')


def detect_requirements_type(requirements_filepath: str) -> str:
    """Infer the requirement type from the contents of a requirements file.

    Scans the file and classifies it as ``'propositional'``, ``'qflra'``, or
    ``'linear'`` based on the tokens it contains (see inline comments for the
    exact detection rules).

    Args:
        requirements_filepath: Path to a ``.txt`` file containing the requirements.

    Returns:
        The detected requirement type as one of ``'propositional'``, ``'qflra'``,
        or ``'linear'``, or None if no requirement type could be detected.
    """
    # Propositional requirements can be written either as Horn rules ('head :- body') or as
    # disjunctive clauses ('y_0 or not y_1'); both are accepted by the propositional parser.
    # The detection order matters: a ':-' token unambiguously marks a propositional Horn rule.
    # Otherwise, QFLRA and linear requirements both contain inequality signs, so we distinguish
    # them by the boolean operators ('or'/'neg') that only QFLRA uses. A clause-style
    # propositional requirement also uses 'or' but, unlike QFLRA, has no inequality sign.
    # An .arff file stores a hierarchical dataset; only the hierarchical backend reads it.
    if requirements_filepath.lower().endswith('.arff'):
        print('Using auto mode ::: Detected hierarchical requirements (ARFF file)!')
        return 'hierarchical'
    inequality_signs = ['>=', '>', '<=', '<']
    # EG: Mihaela please check. Reason: the loop below returned on the FIRST
    # constraint line -- if that line had an inequality but no "or"/"neg", it
    # returned "linear" immediately, so a disjunction ("or"/"neg") appearing on a
    # LATER line was never seen and a QFLRA file was misclassified as linear (the
    # linear backend then silently mishandles the disjunction). Fix: scan every
    # line, record whether any inequality and any boolean op appear, and decide
    # once after the loop.
    has_inequality = has_boolean_op = False
    with open(requirements_filepath, 'r') as f:
        for line in f:
            line = line.strip()
            tokens = line.split()
            # EG: Mihaela please check. Reason: `'ordering' in line` matched the
            # substring "ordering" ANYWHERE in the line, so a constraint line that
            # merely contained that text (a variable name, or a comment like
            # "# reordering") was silently skipped as if it were the ordering
            # declaration. Fix: only skip a line whose first token is exactly "ordering".
            if not line or (tokens and tokens[0] == 'ordering'):
                continue
            if ':-' in tokens:
                print('Using auto mode ::: Detected propositional requirements!')
                return 'propositional'
            if any(sign in line for sign in inequality_signs):
                has_inequality = True
            if 'or' in tokens or 'neg' in tokens:
                has_boolean_op = True
    if has_inequality:
        detected = 'qflra' if has_boolean_op else 'linear'
        print(f'Using auto mode ::: Detected {detected.upper() if detected == "qflra" else detected} requirements!')
        return detected
    if has_boolean_op:
        print('Using auto mode ::: Detected propositional requirements!')
        return 'propositional'
    return None

