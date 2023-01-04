# Implicit Composite Kernel (ICK)

This repository contains PyTorch implementation for the paper "Incorporating Prior Knowledge into Neural Networks through an Implicit Composite Kernel" [1] (ICK) and "Estimating Causal Effect using Multi-task Deep Ensemble" [2] (CMDE, currently under review).

<p align="center">
  <img width="800" alt="Figure_C1" src="https://user-images.githubusercontent.com/45862046/167158033-ff7357c1-5bbd-4a24-9689-f280db1037f2.png">
</p>
<p align="center">
  <em>An illustration of ICK Framework</em>
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
- `models`: This module contains the implementation of the ICK/CMDE frameworks.
  - `ick.py`: Contains the definition of the ICK framework
  - `cmick.py`: Contains the definition of the Causal Multi-task Implicit Composite Kernel (CMICK) framework. Note that based on this definition, CMDE is considered a subset of CMICK since we can easily replace NN with a user-defined kernel as in `kernels/kernel_fn.py`.
- `utils`: This module contains the utility and helper functions used for fitting the ICK model.
  - `constants.py`: Contains constant variables used for other scripts in this directory
  - `data_generator.py`: Contains the definition of data loaders for generating multimodal data
  - `helpers.py`: Contains helper functions (e.g. train-validation-test split)
  - `losses.py`: Contains loss functions used for training CMDE and other benchmark models in causal inference tasks. Note that ICK uses conventional regression loss functions (e.g. MSE loss) for training.
  - `metrics.py`: Contains metrics for model evaluation in causal inference (i.e. CMDE and other benchmark models)
  - `train.py`: Contains trainer classes for fitting ICK (or ICK ensemble) and making predictions

## Tutorial
Please refer to the notebook `tutorial_1d_regression.ipynb` for a detailed tutorial of fitting ICK to multimodal data in a simple regression task. 

## Citation
If you publish any materials using this repository, please include the following Bibtex citation:
```
@article{jiang2022incorporating,
  title={Incorporating Prior Knowledge into Neural Networks through an Implicit Composite Kernel},
  author={Jiang, Ziyang and Zheng, Tongshu and Carlson, David},
  journal={arXiv preprint arXiv:2205.07384},
  year={2022}
}
```

## References
[1]. Jiang, Ziyang, Tongshu Zheng, and David Carlson. "Incorporating Prior Knowledge into Neural Networks through an Implicit Composite Kernel." arXiv preprint arXiv:2205.07384 (2022). <br />
[2]. <br />
[3]. Zhou, Tianhui, William E. Carson IV, and David Carlson. "Estimating potential outcome distributions with collaborating causal networks." arXiv preprint arXiv:2110.01664 (2021). <br />
[4]. Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." Advances in neural information processing systems 30 (2017). <br />
[5]. Shalit, Uri, Fredrik D. Johansson, and David Sontag. "Estimating individual treatment effect: generalization bounds and algorithms." International Conference on Machine Learning. PMLR, 2017. <br />
[6]. Alaa, Ahmed M., and Mihaela Van Der Schaar. "Bayesian inference of individualized treatment effects using multi-task gaussian processes." Advances in neural information processing systems 30 (2017). <br />
[7]. Alaa, Ahmed M., Michael Weisz, and Mihaela Van Der Schaar. "Deep counterfactual networks with propensity-dropout." arXiv preprint arXiv:1706.05966 (2017). <br />
[8]. Hatt, Tobias, and Stefan Feuerriegel. "Estimating average treatment effects via orthogonal regularization." Proceedings of the 30th ACM International Conference on Information & Knowledge Management. 2021. <br />
[9]. Künzel, Sören R., et al. "Metalearners for estimating heterogeneous treatment effects using machine learning." Proceedings of the national academy of sciences 116.10 (2019): 4156-4165. <br />
[10]. Hill, Jennifer L. "Bayesian nonparametric modeling for causal inference." Journal of Computational and Graphical Statistics 20.1 (2011): 217-240. <br />
