# Implicit Composite Kernel (ICK)

This repository presents a more organized and user-friendly PyTorch implementation for the paper "Incorporating Prior Knowledge into Neural Networks through an Implicit Composite Kernel".

<p align="center">
  <img width="800" alt="Figure_C1" src="https://user-images.githubusercontent.com/45862046/167158033-ff7357c1-5bbd-4a24-9689-f280db1037f2.png">
</p>
<p align="center">
  <em>An illustration of ICK Framework</em>
</p>

## Structure of the Repository
The structure of this repository is given below:
- `data`: This directory contains the data for the 3 experiments mentioned in the paper.
- `model_utils`: This directory contains the models and the utility functions for both Implicit Composite Kernel (ICK) and the benchmark neural network-random forest joint model designed by Zheng et al.
  - `cnnrf_utils`: This module contains all the utility functions for running the neural network-random forest joint model.
  - `gp_models`: This module contains the overall definition of the ICK framework.
  - `kernels`: This module contains the implementation details of Nystrom approximation (ICK-y) and Random Fourier Features (ICK-r) as well as the definition of commonly used network architectures and kernel functions, which is the **most important** script in the whole repository.
  - `utils`: This module contains all the utility functions for running the ICK framework, including dataset and dataloader definition, functions for training and testing, etc.
- `experiments`: This directory contains the Jupyter notebooks and Python scripts for running experiments using both synthetic and real-world data.
  - `Synthetic`: Contains Jupyter notebooks for running experiments on synthetic data as described in Section 5.1 of the paper.
  - `Remote_Sensing`: Contains Python scripts for running experiments on remote sensing data as described in Section 5.2 of the paper.
  - `Worker_productivity`: Contains Jupyter notebook for running experiments on the worker productivity data set from the UCI Machine Learning Repository as described in Section 5.3 of the paper.
