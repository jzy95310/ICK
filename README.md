# Implicit Composite Kernel (ICK)

This repository contains PyTorch implementation for the paper "Estimating Causal Effect using Multi-task Deep Ensemble" (CMDE, currently under review).

<p align="center">
  <img width="603" alt="Figure_1" src="https://user-images.githubusercontent.com/45862046/211047479-94c50d8d-0044-4486-92c3-338c060e9464.png">
</p>
<p align="center">
  <em>An illustration of CMDE Framework</em>
</p>

## Structure of the Repository
The structure of this repository is given below:
- `benchmarks`: This module contains the implementation of most of the benchmark models used in ICK and CMDE paper.
  - `ccn.py`: Collaborating Causal Networks
  - `cevae_modified.py`: Causal effect inference with deep latent-variable models, served as a benchmark in CMDE paper
  - `cfrnet.py`: Counterfactual Regression, served as a benchmark in CMDE paper
  - `cmgp_modified.py`: Causal Multi-task Gaussian Process, served as a benchmark in CMDE paper
  - `data_generator.py`: Contains the definition of data loaders for generating multimodal data for `joint_nn` models (e.g. CNN-RF)
  - `dcn_pd.py`: Deep Counterfactual Network with Propensity Dropout, served as a benchmark in CMDE paper
  - `donut.py`: Deep Orthogonal Networks for Unconfounded Treatments, served as a benchmark in CMDE paper
  - `helpers.py`: Contains helper functions for `joint_nn` models (e.g. CNN-RF)
  - `joint_nn.py`: Contains the definition of joint deep neural network models (e.g. CNN-RF)
  - `train_benchmarks.py`: Contains trainer classes for fitting benchmark models in this directory
  - `x_learner.py`: X-learner where the base learner is either random forest (RF) or Bayesian Additive Regression Trees (BART) [10], served as a benchmark in CMDE paper
- `data`: This directory contains all the datasets used for conducting experiments in CMDE paper.
- `experiments`: This directory contains all the scripts needed for replicating the experiments CMDE (see `causal_inference`) paper.
- `kernels`: This module contains the implementation details of Nystrom approximation and Random Fourier Features as well as the definition of commonly used neural network (NN) architectures and kernel functions.
  - `bnn.py`: Contains the definition of NN architectures with **variational** layers
  - `constants.py`: Contains the definition of various activation functions used for building NN architectures
  - `kernel_fn.py`: Contains commonly used kernel functions for both Nystrom approximation and Random Fourier Features implementation
  - `nn.py`: Contains the definition of NN architectures (e.g. dense, convolutional) with **deterministic** layers
  - `nystrom.py`: Contains user-specified implicit kernels based on Nystrom approximation
  - `rff.py`: Contains user-specified implicit kernels based on Random Fourier Features (working in progress)
- `models`: This module contains the implementation of the CMDE frameworks.
  - `cmick.py`: Contains the definition of the Causal Multi-task Deep Ensemble (CMDE) framework.
- `utils`: This module contains the utility and helper functions used for fitting the ICK model.
  - `constants.py`: Contains constant variables used for other scripts in this directory
  - `data_generator.py`: Contains the definition of data loaders for generating multimodal data
  - `helpers.py`: Contains helper functions (e.g. train-validation-test split)
  - `losses.py`: Contains loss functions used for training CMDE and other benchmark models in causal inference tasks. 
  - `metrics.py`: Contains metrics for model evaluation in causal inference (i.e. CMDE and other benchmark models)
  - `train.py`: Contains trainer classes for fitting models
