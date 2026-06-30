"""Propositional requirements subpackage.

Implements PiShield's support for propositional (boolean logic) requirements. Such
requirements are written as Horn rules (``head :- body``) or disjunctive clauses
(``y_0 or not y_1``) over binary variables, and can be enforced either with the
:class:`~pishield.propositional_requirements.shield_layer.ShieldLayer` (which corrects
predictions so the requirements provably hold) or encouraged with the
:class:`~pishield.propositional_requirements.shield_loss.ShieldLoss` (a t-norm based
penalty). The remaining modules provide the supporting logic primitives - literals,
clauses, constraints and their groups - along with stratification, strong-coherency
preprocessing and profiling utilities.
"""
