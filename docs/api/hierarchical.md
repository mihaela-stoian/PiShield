# Hierarchical requirements

The backend for `hierarchical` requirements — class hierarchies (subsumption),
where whenever a class holds all of its ancestors must hold too. It implements
C-HMCNN's Max Constraint Module: each class score is replaced by the maximum
score over its own subtree, guaranteeing hierarchically coherent predictions.
The hierarchy is given as positive Horn rules with a single positive literal in the body (`parent :- child`) or read from an
`.arff` dataset (the FUN/GO format for hierarchical multi-label data).

## Shield Layer

::: pishield.hierarchical_requirements.shield_layer

## Hierarchy

::: pishield.hierarchical_requirements.classes

## Parser

::: pishield.hierarchical_requirements.parser

## Dataset loader

::: pishield.hierarchical_requirements.datasets