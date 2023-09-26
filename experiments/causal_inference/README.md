# Causal Multi-task Deep Ensemble (CMDE)
[![Conference](https://img.shields.io/badge/ICML23-Paper-blue])](https://proceedings.mlr.press/v202/jiang23c/jiang23c.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2301.11351-b31b1b.svg)](https://arxiv.org/abs/2301.11351)

This directory contains the Python scripts of all the experiments in the paper "Estimating Causal Effect using Multi-task Deep Ensemble" [1].

<p align="center">
  <img width="603" alt="Figure_1" src="https://user-images.githubusercontent.com/45862046/211047479-94c50d8d-0044-4486-92c3-338c060e9464.png">
</p>
<p align="center">
  <em>An illustration of CMDE Framework</em>
</p>

## Structure of the Directory

Each file contains one of the experiments, i.e., synthetic, ACIC, Jobs, Twins, COVID, and STAR data, conducted in the CMDE paper as listed below. In each experiment, we compare CMDE (implemented in `model/cmick.py`) with other causal inference models (implemented in `benchmarks`) such as CMGP [2], CEVAE [3], X-learner [4], etc.

## Training and Evaluation of CMDE

Say we have covariates $X$ with dimension of 30 and a binary treatment $T \in {0, 1}$, and we want to predict the potential outcomes $Y_0$ and $Y_1$ for both control and treatment groups using the formulation: $Y_0(X) = \alpha_H f_H(X) + \alpha_{HT} f_{HT}(X)$ and $Y_1(X) = \alpha_{HT} f_{HT}(X) + \alpha_T f_T(X)$ where $f_H$, $f_T$, and $f_{HT}$ are all dense neural networks, then we can define one of the **baselearners** in CMDE as follows:
```
from model import ICK, CMICK

alpha_H, alpha_T, alpha_HT = ...
f_H = ICK(
    kernel_assignment=['ImplicitDenseNetKernel'],
    kernel_params={
        'ImplicitDenseNetKernel':{
            'input_dim': 30,
            'latent_feature_dim': 512,
            'num_blocks': 1, 
            'num_layers_per_block': 1, 
            'num_units': 512, 
            'activation': 'relu'
        }
    }
)
f_T = ICK(
    kernel_assignment=['ImplicitDenseNetKernel'],
    kernel_params={
        'ImplicitDenseNetKernel':{
            'input_dim': 30,
            'latent_feature_dim': 512,
            'num_blocks': 1, 
            'num_layers_per_block': 1, 
            'num_units': 512, 
            'activation': 'relu'
        }
    }
)
f_HT = ICK(
    kernel_assignment=['ImplicitDenseNetKernel'],
    kernel_params={
        'ImplicitDenseNetKernel':{
            'input_dim': 30,
            'latent_feature_dim': 512,
            'num_blocks': 1, 
            'num_layers_per_block': 1, 
            'num_units': 512, 
            'activation': 'relu'
        }
    }
)
baselearner = CMICK(
    control_components=[f_H], treatment_components=[f_T], shared_components=[f_HT],
    control_coeffs=[alpha_H], treatment_coeffs=[alpha_T], shared_coeffs=[alpha_HT], 
    coeff_trainable=True
)
```
Note that we borrow the `ICK` template to define the neural networks for $f_H$, $f_T$, and $f_{HT}$, but users are welcmoe to define their own neural networks as long as it inherits `torch.nn.Module`. After constructing the ensemble, we can also use the `Trainer` classes in `utils.train` to fit it. Please refer to the notebook `synthetic_data_experiment.ipynb` in this directory for an example of fitting CMDE to a synthetic toy dataset in causal inference setting.

## References
[1]. Jiang, Ziyang, et al. "Estimating Causal Effects using a Multi-task Deep Ensemble." arXiv preprint arXiv:2301.11351 (2023). <br />
[2]. Alaa, Ahmed M., and Mihaela Van Der Schaar. "Bayesian inference of individualized treatment effects using multi-task gaussian processes." Advances in neural information processing systems 30 (2017). <br />
[3]. Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." Advances in neural information processing systems 30 (2017). <br />
[4]. Künzel, Sören R., et al. "Metalearners for estimating heterogeneous treatment effects using machine learning." Proceedings of the national academy of sciences 116.10 (2019): 4156-4165. <br />
