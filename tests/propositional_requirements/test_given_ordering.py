import numpy as np

from pishield.propositional_requirements.constraint import Constraint
from pishield.propositional_requirements.literal import Literal
from pishield.shield_layer import build_shield_layer

def test_given_order():
    CL = build_shield_layer(
        41,
        'examples/autonomous_driving/StrongCoherency_CL_ROAD-R/constraints/full',
        requirements_type="propositional")


    assert CL != None