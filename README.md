# Implicit Composite Kernel (ICK)
[![TMLR](https://img.shields.io/badge/TMLR-Paper-blue)](https://openreview.net/pdf?id=HhjSalvWVe)
[![arXiv](https://img.shields.io/badge/arXiv-2205.07384-b31b1b.svg)](https://arxiv.org/abs/2205.07384)

This repository contains PyTorch implementation for the paper "Incorporating Prior Knowledge into Neural Networks through an Implicit Composite Kernel" [1] (ICK, currently under review) and "Estimating Causal Effect using Multi-task Deep Ensemble" [2] (CMDE, accepted by ICML 2023).

<p align="center">
  <img width="800" alt="Figure_C1" src="https://user-images.githubusercontent.com/45862046/167158033-ff7357c1-5bbd-4a24-9689-f280db1037f2.png">
</p>
<p align="center">
  <em>An illustration of ICK Framework</em>
</p>

<p align="center">
  <img width="603" alt="Figure_1" src="https://user-images.githubusercontent.com/45862046/211047479-94c50d8d-0044-4486-92c3-338c060e9464.png">
</p>
<p align="center">
  <em>An illustration of CMDE Framework</em>
</p>

## Structure of the Repository
The structure of this repository is given below:
- `benchmarks`: This module contains the implementation of most of the benchmark models used in ICK and CMDE paper.
  - `ccn.py`: Collaborating Causal Networks [3]
  - `cevae_modified.py`: Causal effect inference with deep latent-variable models [4], served as a benchmark in CMDE paper
  - `cfrnet.py`: Counterfactual Regression [5], served as a benchmark in CMDE paper
  - `cmgp_modified.py`: Causal Multi-task Gaussian Process [6], served as a benchmark in CMDE paper
  - `data_generator.py`: Contains the definition of data loaders for generating multimodal data for `joint_nn` models (e.g. CNN-RF) in ICK paper
  - `dcn_pd.py`: Deep Counterfactual Network with Propensity Dropout [7], served as a benchmark in CMDE paper
  - `donut.py`: Deep Orthogonal Networks for Unconfounded Treatments [8], served as a benchmark in CMDE paper
  - `helpers.py`: Contains helper functions for `joint_nn` models (e.g. CNN-RF) in ICK paper
  - `joint_nn.py`: Contains the definition of joint deep neural network models (e.g. CNN-RF), served as a benchmark in ICK paper
  - `train_benchmarks.py`: Contains trainer classes for fitting benchmark models in this directory
  - `x_learner.py`: X-learner [9] where the base learner is either random forest (RF) or Bayesian Additive Regression Trees (BART) [10], served as a benchmark in CMDE paper
- `data`: This directory contains all the datasets used for conducting experiments in CMDE paper.
- `experiments`: This directory contains all the scripts needed for replicating the experiments in ICK (see `synthetic_data` and `remote_sensing_separable_kernel`) and CMDE (see `causal_inference`) paper.
- `kernels`: This module contains the implementation details of Nystrom approximation (ICK-y) and Random Fourier Features (ICK-r) as well as the definition of commonly used neural network (NN) architectures and kernel functions.
  - `bnn.py`: Contains the definition of NN architectures with **variational** layers
  - `constants.py`: Contains the definition of various activation functions used for building NN architectures
  - `kernel_fn.py`: Contains commonly used kernel functions for both Nystrom approximation and Random Fourier Features implementation
  - `nn.py`: Contains the definition of NN architectures (e.g. dense, convolutional) with **deterministic** layers
  - `nystrom.py`: Contains user-specified implicit kernels based on Nystrom approximation
  - `rff.py`: Contains user-specified implicit kernels based on Random Fourier Features (working in progress)
- `model`: This module contains the implementation of the ICK/CMDE frameworks.
  - `ick.py`: Contains the definition of the ICK framework
  - `cmick.py`: Contains the definition of the Causal Multi-task Implicit Composite Kernel (CMICK) framework. Note that based on this definition, CMDE is considered a subset of CMICK since we can easily replace NN with a user-defined kernel as in `kernels/kernel_fn.py`.
- `utils`: This module contains the utility and helper functions used for fitting the ICK model.
  - `constants.py`: Contains constant variables used for other scripts in this directory
  - `data_generator.py`: Contains the definition of data loaders for generating multimodal data
  - `helpers.py`: Contains helper functions (e.g. train-validation-test split)
  - `losses.py`: Contains loss functions used for training CMDE and other benchmark models in causal inference tasks. Note that ICK uses conventional regression loss functions (e.g. MSE loss) for training.
  - `metrics.py`: Contains metrics for model evaluation in causal inference (i.e. CMDE and other benchmark models)
  - `train.py`: Contains trainer classes for fitting ICK (or ICK ensemble) and making predictions

## Tutorials
### Training and evaluation of ICK
ICK is typically used to fit multi-modal data (e.g. a dataset containing both images and tabular data). To construct an ICK model, we first need to determine the types of kernels we want to use for each modality. Say we want to one kernel implied by a convolutional neural network to process images widh dimension $100 \times 100 \times 3$ and periodic kernel mapped by Nystrom approximation to process the corresponding 1-dimensional timestamp data, then we can define our kernels and ICK model as follows:
```
from model import ICK
from kernels.kernel_fn import periodic_kernel_nys

kernel_assignment = ['ImplicitConvNet2DKernel', 'ImplicitNystromKernel']
kernel_params = {
    'ImplicitConvNet2DKernel':{
        'input_width': 100,
        'input_height': 100, 
        'in_channels': 3, 
        'latent_feature_dim': 16,
        'num_blocks': 2, 
        'num_intermediate_channels': 64, 
        'kernel_size': 3, 
        'stride': 1
    }, 
    'ImplicitNystromKernel': {
        'kernel_func': periodic_kernel_nys, 
        'params': ['std','period','lengthscale','noise'], 
        'vals': [1., 365., 0.5, 0.5], 
        'trainable': [True,True,True,True], 
        'alpha': 1e-5, 
        'num_inducing_points': 16, 
        'nys_space': [[0.,365.]]
    }
}
model = ICK(kernel_assignment, kernel_params)
```
Note that here we also specify the architecture and periodic kernel parameters through the argument `kernel_params`, including the depth (`num_blocks`) and width (`num_intermediate_channels`) of the convolutional network, the filter size (`kernel_size`), and the initial values of those trainable parameters in the periodic kernel including std, period, length scale, and a white noise term. After constructing the model, we can use the `Trainer` classes in `utils.train` to fit it. Please refer to the notebook `tutorial_1d_regression.ipynb` for a more detailed tutorial of fitting ICK (as well as other variants of ICK such as variational ICK, ICK ensemble, etc.) to multi-modal data in a simple regression task. <br />

# Causal Multi-task Deep Ensemble
[![Conference](https://img.shields.io/badge/ICML23-Paper-blue])](https://proceedings.mlr.press/v202/jiang23c/jiang23c.pdf)
[![arXiv](https://img.shields.io/badge/arXiv-2301.11351-b31b1b.svg)](https://arxiv.org/abs/2301.11351)

CMDE/CMICK is typically used for estimating the causal effect of a binary treatment from an observational dataset. Say we have covariates $X$ with dimension of 30 and a binary treatment $T \in {0, 1}$, and we want to predict the potential outcomes $Y_0$ and $Y_1$ for both control and treatment groups using the formulation: $Y_0(X) = \alpha_H f_H(X) + \alpha_{HT} f_{HT}(X)$ and $Y_1(X) = \alpha_{HT} f_{HT}(X) + \alpha_T f_T(X)$ where $f_H$, $f_T$, and $f_{HT}$ are all dense neural networks, then we can define one of the **baselearners** in CMDE as follows:
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
Note that we borrow the `ICK` template to define the neural networks for $f_H$, $f_T$, and $f_{HT}$, but users are welcmoe to define their own neural networks as long as it inherits `torch.nn.Module`. After constructing the ensemble, we can also use the `Trainer` classes in `utils.train` to fit it. Please refer to the notebook `experiments/causal_inference/synthetic_data_experiment.ipynb` for an example of fitting CMDE to a synthetic toy dataset in causal inference setting.

## Citation
If you publish any materials using this repository, please include the following Bibtex citation:
```
@article{jiang2024incorporating,
  title={Incorporating Prior Knowledge into Neural Networks through an Implicit Composite Kernel},
  author={Ziyang Jiang and Tongshu Zheng and Yiling Liu and David Carlson},
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2024},
  url={https://openreview.net/forum?id=HhjSalvWVe}
}

@inproceedings{jiang2023estimating,
  title={Estimating causal effects using a multi-task deep ensemble},
  author={Jiang, Ziyang and Hou, Zhuoran and Liu, Yiling and Ren, Yiman and Li, Keyu and Carlson, David},
  booktitle={International Conference on Machine Learning},
  pages={15023--15040},
  year={2023},
  organization={PMLR}
}
```

## References
[1]. Jiang, Ziyang, Tongshu Zheng, and David Carlson. "Incorporating Prior Knowledge into Neural Networks through an Implicit Composite Kernel." arXiv preprint arXiv:2205.07384 (2022). <br />
[2]. Jiang, Ziyang, et al. "Estimating causal effects using a multi-task deep ensemble." International Conference on Machine Learning. PMLR, 2023. <br />
[3]. Zhou, Tianhui, William E. Carson IV, and David Carlson. "Estimating potential outcome distributions with collaborating causal networks." arXiv preprint arXiv:2110.01664 (2021). <br />
[4]. Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." Advances in neural information processing systems 30 (2017). <br />
[5]. Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment effect: generalization bounds and algorithms." International Conference on Machine Learning. PMLR, 2017. <br />
[6]. Alaa, Ahmed M., and Mihaela Van Der Schaar. "Bayesian inference of individualized treatment effects using multi-task gaussian processes." Advances in neural information processing systems 30 (2017). <br />
[7]. Alaa, Ahmed M., Michael Weisz, and Mihaela Van Der Schaar. "Deep counterfactual networks with propensity-dropout." arXiv preprint arXiv:1706.05966 (2017). <br />
[8]. Hatt, Tobias, and Stefan Feuerriegel. "Estimating average treatment effects via orthogonal regularization." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021. <br />
[9]. Künzel, Sören R., et al. "Metalearners for estimating heterogeneous treatment effects using machine learning." Proceedings of the national academy of sciences 116.10 (2019): 4156-4165. <br />
[10]. Hill, Jennifer L. "Bayesian nonparametric modeling for causal inference." Journal of Computational and Graphical Statistics 20.1 (2011): 217-240. <br />
