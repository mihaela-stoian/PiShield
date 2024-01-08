# CLOVERD: Building Constraint Layers over Deep Neural Networks

## Dependencies
CLOVERD requires PyTorch, NumPy, and Python 3.8 or later.

*Optional step*: conda environment setup, using cpu-only PyTorch here. Different PyTorch versions can be specified following the instructions [here](https://pytorch.org/get-started/locally/).
```
conda create -n "cloverd" python=3.11 ipython
conda activate cloverd
conda install pytorch cpuonly -c pytorch 
pip install numpy
```

## Installation
From the root of this repository, containing `setup.py`, run:
```
pip install .
```

Alternatively, install using pip: (TODO: add package to pip)
```
pip install cloverd
```

## Usage

### Simple Example 1
Assume we have the following constraints and ordering of the variables in a file `example_constraints_tabular.txt`:
```
ordering y_0 y_1 y_2
-y_0 >= -3
y_0 >= 3
y_0 - y_1 >= 0
- y_0 - y_2 >= 0
```

To correct predictions at inference time such that they satisfy the constraints, we can use CLOVERD as follows:
```
from cloverd.constraint_layer import ConstraintLayer

predictions = torch.tensor([[-5., -2., -1.]])
constraints_path = 'example_constraints_tabular.txt'

num_variables = predictions.shape[-1]
CL = ConstraintLayer(num_variables, constraints_path)

# apply CL from CLOVERD to get the corrected predictions
corrected_predictions = CL(predictions.clone())  # returns tensor([[ 3., -2., -3.]], which satisfies the constraints
```
