# PiShield: Building Constraint Layers over Deep Neural Networks

## Dependencies
PiShield requires Python 3.8 or later and PyTorch.

*Optional step*: conda environment setup, using cpu-only PyTorch here. Different PyTorch versions can be specified following the instructions [here](https://pytorch.org/get-started/locally/).
```
conda create -n "pishield" python=3.11 ipython 
conda activate pishield

conda install pytorch cpuonly -c pytorch 
pip install -r requirements.txt
```

## Installation
From the root of this repository, containing `setup.py`, run:
```
pip install .
```

Alternatively, install using pip: (TODO: add package to pip)
```
pip install pishield
```

## Usage

### Simple Example 1: Tabular Data Generation
Assume we have the following constraints and ordering of the variables in a file `example_constraints_tabular.txt`:
```
ordering y_0 y_1 y_2
-y_0 >= -3
y_0 >= 3
y_0 - y_1 >= 0
-y_0 - y_2 >= 0
```

#### Inference time
To correct predictions at inference time such that they satisfy the constraints, we can use PiShield as follows:
```
import torch
from pishield.constraint_layer import build_constraint_layer

predictions = torch.tensor([[-5., -2., -1.]])
constraints_path = 'example_constraints_tabular.txt'

num_variables = predictions.shape[-1]
CL = build_constraint_layer(num_variables, constraints_path)

# apply CL from PiShield to get the corrected predictions
corrected_predictions = CL(predictions.clone())  # returns tensor([[ 3., -2., -3.]], which satisfies the constraints
```

```
import torch
from pishield.constraint_layer import build_constraint_layer

def correct_predictions(predictions: torch.Tensor, constraints_path: str):
    num_variables = predictions.shape[-1]
    
    # build a constraint layer CL using PiShield
    CL = build_constraint_layer(num_variables, constraints_path)
    
    # apply PiShield to get corrected predictions, which satisfy the constraints
    corrected_predictions = CL(predictions)
    return corrected_predictions
```

#### Training time
Assume a Deep Generative Model (DGM) is used to obtain synthetic tabular data.
Using PiShield at training time is easy, as it requires two steps:
1. Instantiating the ConstraintLayer class from PiShield in the DGM's constructor.
2. Applying the ConstraintLayer on the generated data obtained from the DGM before updating the loss function of the DGM.