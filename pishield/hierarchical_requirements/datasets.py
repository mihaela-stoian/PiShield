"""Dataset loader for the hierarchical requirements case.

Loads a hierarchical multi-label classification dataset from an ``.arff`` file
into feature/label arrays aligned with the hierarchical Shield Layer. This is the
data-loading counterpart of :mod:`pishield.hierarchical_requirements.parser`: the
parser reads only the hierarchy from the header, while this module additionally
reads the ``@DATA`` section and returns the features ``X`` and the labels ``Y``.

It reproduces the loading behaviour of C-HMCNN [3] (the FUN/GO functional-genomics
benchmarks): numeric features with ``'?'`` for missing values, categorical
features one-hot encoded, and, crucially, labels expanded up the hierarchy -- when
a class is labelled, all of its ancestors are labelled too, so ``Y`` itself is
hierarchically coherent. The label columns are ordered exactly as
:attr:`Hierarchy.class_names`, i.e. the same order the hierarchical Shield Layer
corrects, so ``Y`` lines up column-for-column with the layer's output.
"""

from typing import List, Tuple

import numpy as np

from pishield.hierarchical_requirements.classes import Hierarchy
from pishield.hierarchical_requirements.parser import parse_arff_hierarchy

# Root classes that are not scored, following C-HMCNN: they carry no information
# (every example belongs to them) and are excluded from evaluation via ``to_eval``.
ROOT_CLASSES_TO_SKIP = ['root', 'GO0003674', 'GO0005575', 'GO0008150']


class HierarchicalDataset:
    """A loaded hierarchical dataset: features, labels and the class hierarchy.

    Attributes:
        X: The features, shape (num_samples, num_features), missing values imputed.
        Y: The binary labels, shape (num_samples, num_classes), ancestor-closed and
            column-aligned with :attr:`Hierarchy.class_names`.
        hierarchy: The parsed :class:`Hierarchy` (shared by all splits of a dataset).
        to_eval: Boolean mask of shape (num_classes,), False for the root classes
            that are excluded from evaluation (see :data:`ROOT_CLASSES_TO_SKIP`).
    """

    def __init__(self, X: np.ndarray, Y: np.ndarray, hierarchy: Hierarchy, to_eval: np.ndarray):
        """Store the loaded arrays and hierarchy."""
        self.X = X
        self.Y = Y
        self.hierarchy = hierarchy
        self.to_eval = to_eval

    @property
    def class_names(self) -> List[str]:
        """The class names in label-column order."""
        return self.hierarchy.class_names

    @property
    def num_classes(self) -> int:
        """The number of classes (label columns)."""
        return self.hierarchy.num_classes

    @property
    def num_features(self) -> int:
        """The number of feature columns."""
        return self.X.shape[1]

    def __repr__(self):
        """Return a short summary of the dataset's shape."""
        return (f'HierarchicalDataset(samples={self.X.shape[0]}, features={self.num_features}, '
                f'classes={self.num_classes}, evaluated_classes={int(self.to_eval.sum())})')


def load_arff_dataset(filepath: str, arff_hierarchy_style: str = 'auto',
                      impute_missing: bool = True) -> HierarchicalDataset:
    """Load a hierarchical dataset from an ``.arff`` file.

    Reads the hierarchy from the header (via :func:`parse_arff_hierarchy`, so the
    class ordering matches the hierarchical Shield Layer) and then the ``@DATA``
    rows into features and ancestor-closed labels.

    Args:
        filepath: Path to the ``.arff`` dataset file.
        arff_hierarchy_style: The hierarchy format: ``'auto'`` (default), ``'paths'``
            or ``'edges'`` (see :func:`parse_arff_hierarchy`).
        impute_missing: If True, replace missing numeric values (``'?'`` -> NaN)
            with their column mean, as in C-HMCNN. If False, leave them as NaN.

    Returns:
        A :class:`HierarchicalDataset` with ``X``, ``Y``, the ``hierarchy`` and the
        ``to_eval`` mask.

    Raises:
        Exception: If a data row has no label field.

    Example:
        >>> data = load_arff_dataset('cellcycle_FUN.train.arff')
        >>> data.X.shape, data.Y.shape
        ((1628, 77), (1628, 500))
    """
    hierarchy = parse_arff_hierarchy(filepath, arff_hierarchy_style)
    name_to_index = {name: index for index, name in enumerate(hierarchy.class_names)}
    num_classes = hierarchy.num_classes

    feature_specs = _parse_feature_specs(filepath)
    num_feature_fields = len(feature_specs)

    X_rows, Y_rows = [], []
    reading_data = False
    with open(filepath, 'r') as f:
        for line in f:
            if not reading_data:
                if line.strip().upper().startswith('@DATA'):
                    reading_data = True
                continue
            # Drop trailing comments and surrounding whitespace; skip blank lines.
            row = line.split('%')[0].strip()
            if not row:
                continue
            fields = row.split(',')
            if len(fields) <= num_feature_fields:
                raise Exception(f'Data row in {filepath} has no label field: "{row}"')

            X_rows.append(_parse_feature_row(fields[:num_feature_fields], feature_specs))
            Y_rows.append(_parse_label_field(fields[num_feature_fields].strip(),
                                             name_to_index, hierarchy, num_classes))

    X = np.array(X_rows, dtype=float)
    Y = np.stack(Y_rows)

    if impute_missing:
        X = _impute_column_mean(X)

    to_eval = np.array([name not in ROOT_CLASSES_TO_SKIP for name in hierarchy.class_names])
    return HierarchicalDataset(X, Y, hierarchy, to_eval)


def _parse_feature_specs(filepath: str) -> List[Tuple[str, List[str]]]:
    """Read the feature attributes from an ARFF header, in order.

    Every ``@ATTRIBUTE`` line before ``@DATA`` except the hierarchical class
    attribute is a feature. Each is recorded as ``('numeric', None)`` or
    ``('categorical', [categories])``.

    Args:
        filepath: Path to the ``.arff`` file.

    Returns:
        The feature specifications in header order.
    """
    specs = []
    with open(filepath, 'r') as f:
        for line in f:
            stripped = line.strip()
            if stripped.upper().startswith('@DATA'):
                break
            if not stripped.upper().startswith('@ATTRIBUTE') or 'hierarchical' in stripped:
                continue
            tokens = stripped.split()
            feature_type = tokens[2] if len(tokens) >= 3 else 'numeric'
            if feature_type.lower() == 'numeric':
                specs.append(('numeric', None))
            else:
                categories = feature_type.strip('{}').split(',')
                specs.append(('categorical', categories))
    return specs


def _parse_feature_row(values: List[str], feature_specs: List[Tuple[str, List[str]]]) -> List[float]:
    """Parse one row's feature fields into a flat numeric vector.

    Numeric features become a single value (``NaN`` for ``'?'``); categorical
    features are one-hot encoded (all-zero for missing or unknown categories).

    Args:
        values: The raw feature fields of a data row.
        feature_specs: The feature specifications from :func:`_parse_feature_specs`.

    Returns:
        The concatenated feature values for the row.
    """
    row = []
    for (kind, categories), value in zip(feature_specs, values):
        value = value.strip()
        if kind == 'numeric':
            row.append(float(value) if value != '?' else np.nan)
        else:
            one_hot = [0.0] * len(categories)
            if value in categories:
                one_hot[categories.index(value)] = 1.0
            row.extend(one_hot)
    return row


def _parse_label_field(label_field: str, name_to_index: dict,
                       hierarchy: Hierarchy, num_classes: int) -> np.ndarray:
    """Parse one row's label field into an ancestor-closed binary label vector.

    The label field is an ``@``-separated list of class names. Each name (with
    ``/`` mapped to ``.`` to match path-style class identities) sets its own column
    and all of its ancestors' columns to 1.

    Args:
        label_field: The raw label field of a data row.
        name_to_index: Mapping from class name to variable index.
        hierarchy: The parsed hierarchy (for ancestor lookup).
        num_classes: The number of classes (length of the label vector).

    Returns:
        The binary label vector of shape (num_classes,).
    """
    y = np.zeros(num_classes)
    for token in label_field.split('@'):
        name = token.strip().replace('/', '.')
        if not name:
            continue
        index = name_to_index.get(name)
        if index is None:
            continue
        y[index] = 1.0
        for ancestor in hierarchy.ancestors(index):
            y[ancestor] = 1.0
    return y


def _impute_column_mean(X: np.ndarray) -> np.ndarray:
    """Replace NaN entries with their column mean (all-NaN columns become 0).

    Args:
        X: The feature array, possibly containing NaN.

    Returns:
        The array with missing values imputed. Returned unchanged if it has no NaN.
    """
    if not np.isnan(X).any():
        return X
    column_mean = np.nanmean(X, axis=0)
    column_mean = np.where(np.isnan(column_mean), 0.0, column_mean)
    nan_rows, nan_cols = np.where(np.isnan(X))
    X[nan_rows, nan_cols] = np.take(column_mean, nan_cols)
    return X
