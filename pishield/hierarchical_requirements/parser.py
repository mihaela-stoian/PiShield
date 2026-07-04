"""Parser for hierarchical requirements files.

Reads a hierarchy of classes into a :class:`Hierarchy` from either of two formats:

* A standard PiShield requirements ``.txt`` file with one Horn rule
  ``parent :- child`` per line (a syntactic subset of the propositional format),
  optionally preceded by an ``ordering`` line naming the variables.
* An ``.arff`` file (the common storage format of hierarchical multi-label
  datasets, e.g. the FUN/GO benchmarks): only the header is read, since the whole
  hierarchy is stored on the single ``@ATTRIBUTE class hierarchical ...`` line.
  The data section is never loaded.

An ARFF hierarchy is written in one of two formats, selected with
``arff_hierarchy_style`` (``'auto'`` by default, which detects the format from the
file itself):

* ``'paths'`` (FUN-style, the CLUS/HMC convention): the hierarchy is written as full
  root-to-node paths, e.g. ``1/2/3``; class identity is the cumulative dotted prefix
  (``1``, ``1.2``, ``1.2.3``) and an implicit ``root`` class is added.
* ``'edges'`` (GO-style): the hierarchy is written as ``parent/child`` edges, e.g.
  ``GO0001/GO0003``; each branch is one edge between named classes.

Both formats describe the same kind of object (a class hierarchy, in general a DAG);
they differ only in how it is serialised. In both, the class-to-variable-index
mapping reproduces C-HMCNN's deterministic ordering (sorted by depth in the
hierarchy, then by name), so labels produced with the C-HMCNN/AWX tooling line up
with the parsed variables.
"""

from typing import List, Tuple

import networkx as nx

from pishield.hierarchical_requirements.classes import Hierarchy

# The two ARFF serialisation formats; 'auto' (the default) detects which one a file uses.
# 'paths' files list full root-to-node paths (e.g. FUN); 'edges' files list parent/child edges (e.g. GO).
ARFF_HIERARCHY_STYLES = ['paths', 'edges']

# A single parent/child edge has two endpoints, so every branch of an 'edges' file has two terms.
TERMS_PER_EDGE = 2


def detect_arff_hierarchy_style(branches: List[str]) -> str:
    """Detect which format an ARFF class hierarchy is written in.

    The hierarchy on an ``@ATTRIBUTE class hierarchical ...`` line is a
    comma-separated list of branches, serialised in one of two formats:
    ``'paths'``, in which each branch is a full root-to-class path (e.g.
    ``"01/02/05"``), or ``'edges'``, in which each branch is a single
    ``parent/child`` edge (e.g. ``"GO0001/GO0003"``). Because an edge has two
    endpoints, every branch of an ``'edges'`` file has exactly two terms, whereas a
    ``'paths'`` file has branches of varying length (its top-level classes are a
    single term); the formats are distinguished on this basis. When every branch has
    two terms the two readings coincide -- each branch is a single edge either way --
    so the detected format is unambiguous.

    Args:
        branches: The branch strings of the hierarchy attribute, i.e. the
            comma-separated pieces of its value.

    Returns:
        The detected format, ``'edges'`` or ``'paths'``.
    """
    all_two_terms = bool(branches) and all(len(branch.split('/')) == TERMS_PER_EDGE for branch in branches)
    return 'edges' if all_two_terms else 'paths'


def parse_hierarchy_file(filepath: str, arff_hierarchy_style: str = 'auto') -> Hierarchy:
    """Parse a hierarchical requirements file into a :class:`Hierarchy`.

    Dispatches on the file extension: ``.arff`` files are parsed header-only via
    :func:`parse_arff_hierarchy`, anything else is parsed as a Horn-rule
    requirements file via :func:`parse_horn_requirements_file`.

    Args:
        filepath: Path to the requirements file (``.txt`` Horn rules or ``.arff``).
        arff_hierarchy_style: The format of ``.arff`` files: ``'auto'`` (the
            default) detects it from the file, or force ``'paths'`` (FUN-style full
            paths) or ``'edges'`` (GO-style parent/child pairs). Ignored for
            non-ARFF files.

    Returns:
        The parsed :class:`Hierarchy`.
    """
    if filepath.lower().endswith('.arff'):
        return parse_arff_hierarchy(filepath, arff_hierarchy_style)
    return parse_horn_requirements_file(filepath)


def parse_horn_requirements_file(filepath: str) -> Hierarchy:
    """Parse a Horn-rule requirements file into a :class:`Hierarchy`.

    Each line must be a rule ``parent :- child`` whose head and body are a single
    positive literal each: a variable index (e.g. ``3``), optionally with the
    ``y_`` prefix used elsewhere in PiShield (e.g. ``y_3``). An optional
    ``ordering`` line names the variables and fixes their number; without it,
    variables are ``0..max_index`` seen in the rules.

    Args:
        filepath: Path to the requirements file.

    Returns:
        The parsed :class:`Hierarchy`.

    Raises:
        Exception: If a line is not a valid hierarchy rule (negated literal,
            more than one body literal, malformed rule), if a rule references a
            variable outside the ordering, or if the edges contain a cycle.
    """
    class_names = None
    edges = []
    max_index = -1

    with open(filepath, 'r') as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            if line.startswith('ordering'):
                class_names = line.split()[1:]
                continue

            tokens = line.split()
            # Tolerate the propositional '<weight> head :- body' variant by dropping the leading token.
            if len(tokens) >= 3 and tokens[2] == ':-':
                tokens = tokens[1:]
            if len(tokens) != 3 or tokens[1] != ':-':
                raise Exception(f'Line {line_number} of {filepath} is not a hierarchy rule "parent :- child" '
                                f'(a single positive literal on each side): "{line}". '
                                f'For general propositional rules use requirements_type="propositional".')

            parent = _parse_positive_literal(tokens[0], line_number, filepath)
            child = _parse_positive_literal(tokens[2], line_number, filepath)
            edges.append((child, parent))
            max_index = max(max_index, child, parent)

    if class_names is None:
        class_names = [str(index) for index in range(max_index + 1)]
    elif max_index >= len(class_names):
        raise Exception(f'The rules in {filepath} reference variable {max_index}, '
                        f'but the ordering line only declares {len(class_names)} variables!')

    return Hierarchy(class_names, edges)


def _parse_positive_literal(token: str, line_number: int, filepath: str) -> int:
    """Parse a positive literal token into a variable index.

    Args:
        token: The literal token, e.g. ``'3'`` or ``'y_3'``.
        line_number: The line the token appears on (for error messages).
        filepath: The file being parsed (for error messages).

    Returns:
        The variable index.

    Raises:
        Exception: If the literal is negated (``n``/``not`` prefix) or is not a
            variable index.
    """
    stripped = token[2:] if token.startswith('y_') else token
    if token.startswith('n') or token.startswith('not'):
        raise Exception(f'Line {line_number} of {filepath} contains the negated literal "{token}": '
                        f'hierarchy rules must be positive ("parent :- child"). '
                        f'For rules with negation use requirements_type="propositional".')
    try:
        index = int(stripped)
    except ValueError:
        raise Exception(f'Line {line_number} of {filepath} contains the literal "{token}", '
                        f'which is not a variable index (e.g. "3" or "y_3").')
    if index < 0:
        raise Exception(f'Line {line_number} of {filepath} contains the negative variable index "{token}"!')
    return index


def parse_arff_hierarchy(filepath: str, arff_hierarchy_style: str = 'auto') -> Hierarchy:
    """Parse the hierarchy from an ARFF file header into a :class:`Hierarchy`.

    Only the header is read: the function scans for the
    ``@ATTRIBUTE class hierarchical ...`` line and stops at ``@DATA``, so the
    (potentially large) data section is never loaded.

    Args:
        filepath: Path to the ``.arff`` file.
        arff_hierarchy_style: ``'auto'`` (the default) detects the format from the
            branches via :func:`detect_arff_hierarchy_style`; ``'paths'`` forces
            FUN-style full paths from the root (with an implicit ``root`` class);
            ``'edges'`` forces GO-style ``parent/child`` pairs.

    Returns:
        The parsed :class:`Hierarchy`, with classes indexed in C-HMCNN's
        deterministic order (sorted by depth in the hierarchy, then by name).

    Raises:
        Exception: If the format is unknown, the file has no hierarchical class
            attribute, or (in ``'edges'`` format) a branch is not a single
            ``parent/child`` pair.
    """
    if arff_hierarchy_style not in ARFF_HIERARCHY_STYLES + ['auto']:
        raise Exception(f'Unknown ARFF hierarchy style "{arff_hierarchy_style}"! '
                        f'Choose "auto" or one of {ARFF_HIERARCHY_STYLES}.')

    hierarchy_spec = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.upper().startswith('@DATA'):
                break
            if line.upper().startswith('@ATTRIBUTE') and 'hierarchical' in line:
                hierarchy_spec = line.split('hierarchical')[1].strip()
                break
    if hierarchy_spec is None:
        raise Exception(f'No "@ATTRIBUTE class hierarchical ..." line found in the header of {filepath}!')

    branches = [branch.strip() for branch in hierarchy_spec.split(',') if branch.strip()]

    if arff_hierarchy_style == 'auto':
        arff_hierarchy_style = detect_arff_hierarchy_style(branches)
        print(f'Auto-detected "{arff_hierarchy_style}" ARFF hierarchy format for {filepath}.')

    # Build the name graph with edges pointing descendant -> ancestor, as in C-HMCNN.
    graph = nx.DiGraph()
    if arff_hierarchy_style == 'paths':
        # 'paths' format: each branch is a full root-to-node path.
        for branch in branches:
            terms = branch.split('/')
            if len(terms) == 1:
                graph.add_edge(terms[0], 'root')
            else:
                for i in range(2, len(terms) + 1):
                    graph.add_edge('.'.join(terms[:i]), '.'.join(terms[:i - 1]))
        # C-HMCNN's FUN ordering: sort by path depth, then name.
        nodes = sorted(graph.nodes(), key=lambda name: (len(name.split('.')), name))
    else:
        # 'edges' format: each branch is one parent/child pair.
        for branch in branches:
            terms = branch.split('/')
            if len(terms) != TERMS_PER_EDGE:
                raise Exception(f'Branch "{branch}" of {filepath} is not a "parent/child" pair: '
                                f'with arff_hierarchy_style="edges" every branch must have exactly two terms.')
            parent, child = terms
            graph.add_edge(child, parent)
        nodes = sorted(graph.nodes(), key=lambda name: (_depth_to_top(graph, name), name))

    nodes_index = {name: index for index, name in enumerate(nodes)}
    edges = [(nodes_index[child], nodes_index[parent]) for child, parent in graph.edges()]
    return Hierarchy(nodes, edges)


def _depth_to_top(graph: nx.DiGraph, name: str) -> int:
    """Return the depth of a class: its shortest distance to a top-level class.

    Reproduces C-HMCNN's GO ordering, which sorts by the shortest path to the
    ``root`` class; when no class is named ``root``, the shortest distance to
    any class without parents is used instead.

    Args:
        graph: The name graph, with edges pointing descendant -> ancestor.
        name: The class name.

    Returns:
        The depth of the class in the hierarchy. Falls back to 0 when no top-level
        class is reachable (e.g. a degenerate, cyclic graph), so ordering never
        fails; a genuinely cyclic graph is then reported by :class:`Hierarchy`.
    """
    if 'root' in graph:
        return nx.shortest_path_length(graph, name, 'root')
    tops = [node for node in graph.nodes() if graph.out_degree(node) == 0]
    distances = [nx.shortest_path_length(graph, name, top)
                 for top in tops if nx.has_path(graph, name, top)]
    return min(distances) if distances else 0


def arff_to_requirements(arff_filepath: str, output_filepath: str,
                         arff_hierarchy_style: str = 'auto') -> Hierarchy:
    """Convert an ARFF hierarchy into a standard requirements ``.txt`` file.

    Writes one ``parent :- child`` rule per direct edge, over variable indices.
    The output contains rules only (no ``ordering`` line), so it is also a valid
    *propositional* requirements file and can be used with
    ``requirements_type='propositional'``. The class-name-to-index mapping is
    available on the returned hierarchy as ``class_names``.

    Args:
        arff_filepath: Path to the ``.arff`` file.
        output_filepath: Path of the requirements file to write.
        arff_hierarchy_style: ``'auto'``, ``'paths'`` or ``'edges'`` (see
            :func:`parse_arff_hierarchy`).

    Returns:
        The parsed :class:`Hierarchy` that was written out.
    """
    hierarchy = parse_arff_hierarchy(arff_filepath, arff_hierarchy_style)
    with open(output_filepath, 'w') as f:
        f.write('\n'.join(hierarchy.to_requirements_lines()) + '\n')
    return hierarchy
