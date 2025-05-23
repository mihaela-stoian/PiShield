# PiShield: A NeSy Framework for Learning with Requirements


* :sparkles: [Description](#sparkles-description)
* :pushpin: [Dependencies](#pushpin-dependencies)
* :hammer_and_wrench: [Installation](#hammer_and_wrench-installation)
* :bulb: [Usage](#bulb-usage)
    - [Inference time](#inference-time)
    - [Training time](#training-time)
* :arrow_forward: [Demo video](#arrow_forward-demo-video)
* :fire: [Performance](#fire-performance)
  + [1. Autonomous Driving](#1-autonomous-driving)
  + [2. Tabular Data Generation](#2-tabular-data-generation)
  + [3. Functional Genomics](#3-functional-genomics)
* [Team](#team)
* :memo: [References](#memo-references)


**Update**: DRL [4] is now part of PiShield and is available on the `drl` branch of this repository!

## :sparkles: Description

Update: PiShield's **website** is now available [here](https://sites.google.com/view/pishield).

PiShield is the first framework ever allowing for the integration of the requirements into the neural networks' topology.

:white_check_mark: The integration happens in a straightforward and efficient manner and allows for the creation of deep learning models that are guaranteed to be compliant with the given requirements, no matter the input.

:pencil2: The requirements can be integrated both at inference and/or training time, depending on the practitioners' needs.

<img height="240" src="./extra/overview.png" width="400"/>

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

## :bulb: Usage

Assume we have the following constraints and ordering of the variables in a file `example_constraints_tabular.txt`:
```
ordering y_0 y_1 y_2
-y_0 >= -3
y_0 >= 3
y_0 - y_1 >= 0
-y_0 - y_2 >= 0
```

### Inference time
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

### Training time
Assume a Deep Generative Model (DGM) is used to obtain synthetic tabular data.
Using PiShield at training time is easy, as it requires two steps:
1. Instantiating the ConstraintLayer class from PiShield in the DGM's constructor.
2. Applying the ConstraintLayer on the generated data obtained from the DGM before updating the loss function of the DGM.


## :arrow_forward: Demo video

A demo video is available for download [here](https://github.com/mihaela-stoian/PiShield/blob/main/extra/video.mp4).


## :fire: Performance

### 1. Autonomous Driving

In [2], we considered standard 3D-RetinaNet models with different temporal learning architectures such as I3D, C2D, RCN, RCGRU, RCLSTM, and SlowFast, and compared each of these with their constrained versions.
The constrained versions inject propositional background knowledge into the models via a constrained layer, as we call it in [2], which is equivalent to a Shield layer when using PiShield.

Below we report the aggregated performance from Table 2 of our paper [2] to show the results we obtained according to the f-mAP (framewise mean Average Precision) measure, at IOU (Intersection-over-Union) threshold of 0.5. 
The best results are in **bold**.

As we can see, the models incorporating background knowledge through **Shield layers** outperform their standard counterparts.

| 	          | Baseline 	 | <span style="color:darkgreen">Shielded</span>  	 |
|------------|------------|--------------------------------------------------|
| I3D      	 | 29.30    	 | **30.98**     	                                  |
| C2D      	 | 26.34    	 | **27.93**     	                                  |
| RCN      	 | 29.26    	 | **30.02**     	                                  |
| RCGRU    	 | 29.24    	 | **30.50**     	                                  |
| RCLSTM   	 | 28.93    	 | **30.42**     	                                  |
| SlowFast 	 | 29.73    	 | **31.88**     	                                  |
|___________ |
| AVERAGE    | 	28.80	    | **30.29**	                                       |																						


### 2. Tabular Data Generation

In [1], we compared standard deep generative models with their respective constrained versions, which use linear inequality constraints.
The latter are the models to which we added a constraint layer, as we call it in [1], which is equivalent to a Shield layer when using PiShield.

Below we reproduce Table 2 of our paper [1] to show the results we obtained according to two standard measures for tabular data generation benchmarks: utility and detection.
For each of these two measures, we report the performance using three metrics: F1-score (F1), weighted F1-score (wF1), and Area Under the ROC Curve (AUC).
The best results are in bold.

As we can see, in 28/30 cases, the models incorporating background knowledge through **Shield layers** outperform their standard counterparts.

|                                                    |            | Utility(**&uarr;**) |             |            | Detection(**&darr;**) |           |               
|----------------------------------------------------|------------|---------------------|-------------|------------|--------------------|-----------|
|                                                    | F1         | wF1                 | AUC         | F1         | wF1                | AUC       |
| WGAN                                               | 0.463      | 0.488               | 0.730       | 0.945      | 0.943              | 0.954     |
| <span style="color:green">Shielded-WGAN</span>     | **0.483**  | **0.502**           | **0.745**   | **0.915**  | **0.912**          | **0.934** |
| TableGAN                                           | 0.330      | 0.400               | 0.704       | 0.908      | 0.907              | 0.926     |
| <span style="color:green">Shielded-TableGAN</span> | **0.375**  | **0.432**           | **0.714**   | **0.898**  | **0.895**          | **0.917** |
| CTGAN                                              | **0.517**  | 0.532               | 0.771       | 0.902      | 0.901              | 0.920     |
| <span style="color:green">Shielded-CTGAN</span>    | 0.516      | **0.537**           | **0.773**   | **0.894**  | **0.891**          | **0.919** |
| TVAE                                               | 0.497      | 0.527               | 0.767       | 0.869      | 0.868              | **0.892** |
| <span style="color:green">Shielded-TVAE</span>     | **0.507**  | **0.537**           | **0.773**   | **0.868**  | **0.867**          | 0.898     |
| GOGGLE                                             | 0.344      | 0.373               | 0.624       | 0.926      | 0.926              | 0.943     |
| <span style="color:green">Shielded-GOGGLE</span>   | **0.409**  | **0.427**           | **0.667**   | **0.925**  | **0.916**          | **0.937** |
| ________________                                   |
| AVERAGE Baseline	                                  | 0.430	     | 0.464               | 	0.719	     | 0.910	     | 0.909              | 	0.927    |
| AVERAGE Shielded 	                                 | **0.458**	 | **0.487**           | 	**0.734**	 | **0.900**	 | **0.896**	         | **0.921** |

### 3. Functional Genomics

In our paper [3], we compared the performance of baseline models with their constrained counterparts.
The latter are the models to which we added a constraint layer, as we call it in [3], which is equivalent to a Shield layer when using PiShield.

Below we aggregated the results from Table 3 of [3] and reported the performance in terms of the area under the average precision and recall curve (AU(PRC)), which is a standard metric used in functional genomics.
As we can see, the models **Shield layers** outperform their standard counterparts.


| Dataset    | Baseline* | PiShield  |
|------------|-----------|-----------|
| CELLCYCLE  | 0.220     | **0.232** |
| DERISI     | 0.179     | **0.182** |
| EISEN      | 0.262     | **0.285** |
| EXPRE      | 0.246     | **0.270** |
| GASCH1     | 0.239     | **0.261** |
| GASCH2     | 0.221     | **0.235** |
| SEQ        | 0.245     | **0.274** |
| SPO        | 0.186     | **0.190** |
| __________ |
| AVERAGE    | 0.225     | **0.241** |

*Note: All baselines for the functional genomics scenario have a postprocessing step included, as functional genomics tasks always require that the constraints are satisfied.


## Team

Mihaela Catalina Stoian [[GitHub]](https://github.com/mihaela-stoian)[[LinkedIn]](https://www.linkedin.com/in/mihaela-catalina-stoian-919b27bb/)[[Google Scholar]](https://scholar.google.com.co/citations?user=B_48apwAAAAJ&hl=en)\
Alex Tatomir [[GitHub]](https://github.com/atatomir)[[LinkedIn]](https://www.linkedin.com/in/atatomir/)[[Google Scholar]](https://scholar.google.com/citations?user=DrGgfBwAAAAJ)\
Thomas Lukasiewicz [[Google Scholar]](https://scholar.google.co.uk/citations?user=arjucpEAAAAJ&hl=en)\
Eleonora Giunchiglia [[GitHub]](https://github.com/EGiunchiglia)[[LinkedIn]](https://www.linkedin.com/in/eleonora-giunchiglia-3063b5164/)[[Google Scholar]](https://scholar.google.com/citations?user=HAgGqScAAAAJ&hl=it)

## :memo: References

[1] Mihaela Catalina Stoian, Salijona Dyrmishi, Maxime Cordy, Thomas Lukasiewicz, Eleonora Giunchiglia. How Realistic Is Your Synthetic Data? Constraining Deep Generative Models for Tabular Data. arXiv:2402.04823. In Proc. of International
Conference on Learning Representations (ICLR), 2024.

[2] Eleonora Giunchiglia, Alex Tatomir, Mihaela Catalina Stoian, Thomas Lukasiewicz. CCN+: A neuro-symbolic framework for deep learning with requirements. International Journal of Approximate Reasoning, 2024.

[3] Eleonora Giunchiglia and Thomas Lukasiewicz. Coherent Hierarchical Multi-Label Classification Networks. In Proceedings of Neural
Information Processing Systems, 2020.

[4] Mihaela Catalina Stoian and Eleonora Giunchiglia. Beyond the Convexity Assumption: Realistic Tabular Data Generation under Quantifier-Free Real Linear Constraints. In Proc. of International Conference on Learning Representations (ICLR) 2025.

