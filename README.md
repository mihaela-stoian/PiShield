# PiShield: a NeSy Framework for Learning with Requirements

## :sparkles: Description

PiShield is the first framework ever allowing for the integration of the requirements into the neural networks' topology.

:white_check_mark: The integration happens in a straightforward and efficient manner and allows for the creation of deep learning models that are guaranteed to be compliant with the given requirements, no matter the input.

:pencil2: The requirements can be integrated both at inference and/or training time, depending on the practitioners' needs.


## :pushpin: Dependencies
PiShield requires Python 3.8 or later and PyTorch.

*Optional step*: conda environment setup, using cpu-only PyTorch here. Different PyTorch versions can be specified following the instructions [here](https://pytorch.org/get-started/locally/).
```
conda create -n "pishield" python=3.11 ipython 
conda activate pishield

conda install pytorch cpuonly -c pytorch 
pip install -r requirements.txt
```

## :hammer_and_wrench: Installation
From the root of this repository, containing `setup.py`, run:
```
pip install .
```

Alternatively, install using pip: (TODO: add package to pip)
```
pip install pishield
```

## :bulb: Usage

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
from pishield.constraint_layer import build_pishield_layer

predictions = torch.tensor([[-5., -2., -1.]])
constraints_path = 'example_constraints_tabular.txt'

num_variables = predictions.shape[-1]
CL = build_pishield_layer(num_variables, constraints_path)

# apply CL from PiShield to get the corrected predictions
corrected_predictions = CL(predictions.clone())  # returns tensor([[ 3., -2., -3.]], which satisfies the constraints
```

```
import torch
from pishield.constraint_layer import build_pishield_layer

def correct_predictions(predictions: torch.Tensor, constraints_path: str):
    num_variables = predictions.shape[-1]
    
    # build a constraint layer CL using PiShield
    CL = build_pishield_layer(num_variables, constraints_path)
    
    # apply PiShield to get corrected predictions, which satisfy the constraints
    corrected_predictions = CL(predictions)
    return corrected_predictions
```

#### Training time
Assume a Deep Generative Model (DGM) is used to obtain synthetic tabular data.
Using PiShield at training time is easy, as it requires two steps:
1. Instantiating the ConstraintLayer class from PiShield in the DGM's constructor.
2. Applying the ConstraintLayer on the generated data obtained from the DGM before updating the loss function of the DGM.

## :fire: Performance


### Autonomous Driving

Excerpt from Table 2 of our paper [2]. 

|          	| Baseline 	| Corrected 	|
|----------	|----------	|-----------	|
| I3D      	| 29.30    	| 30.37     	|
| C2D      	| 26.34    	| 27.93     	|
| RCN      	| 29.26    	| 30.02     	|
| RCGRU    	| 29.24    	| 30.50     	|
| RCLSTM   	| 28.93    	| 29.91     	|
| SlowFast 	| 29.73    	| 31.88     	|


### Tabular Data Generation

Reproduction of Table 2 of our paper [1].

|            | F1    | wF1   | AUC   | F1    | wF1   | AUC   |
|------------|-------|-------|-------|-------|-------|-------|
| WGAN       | 0.463 | 0.488 | 0.730 | 0.945 | 0.943 | 0.954 |
| C-WGAN     | **0.483** | **0.502** | **0.745** | **0.915** | **0.912** | **0.934** |
| TableGAN   | 0.330 | 0.400 | 0.704 | 0.908 | 0.907 | 0.926 |
| C-TableGAN | **0.375** | **0.432** | **0.714** | **0.898** | **0.895** | **0.917** |
| CTGAN      | **0.517** | 0.532 | 0.771 | 0.902 | 0.901 | 0.920 |
| C-CTGAN    | 0.516 | **0.537** | **0.773** | **0.894** | **0.891** | **0.919** |
| TVAE       | 0.497 | 0.527 | 0.767 | 0.869 | 0.868 | **0.892** |
| C-TVAE     | **0.507** | **0.537** | **0.773** | **0.868** | **0.867** | 0.898 |
| GOGGLE     | 0.344 | 0.373 | 0.624 | 0.926 | 0.926 | 0.943 |
| C-GOGGLE   | **0.409** | **0.427** | **0.667** | **0.925** | **0.916** | **0.937** |


### Functional Genomics

Reproduction of Table TODO of our paper [2].



```
@inproceedings{stoian2024cdgm,
title={How Realistic Is Your Synthetic Data? Constraining Deep Generative Models for Tabular Data},
author={Mihaela Catalina Stoian and Salijona Dyrmishi and Maxime Cordy and Thomas Lukasiewicz and Eleonora Giunchiglia},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=tBROYsEz9G}
}

@article{giunchiglia2024ccn_plus,
title = {CCN+: A neuro-symbolic framework for deep learning with requirements},
journal = {International Journal of Approximate Reasoning},
pages = {109124},
year = {2024},
url = {https://www.sciencedirect.com/science/article/pii/S0888613X24000112},
author = {Eleonora Giunchiglia and Alex Tatomir and Mihaela Catalina Stoian and Thomas Lukasiewicz},
}
```
