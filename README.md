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
- `kernels`: This module contains the implementation details of Nystrom approximation (ICK-y) and Random Fourier Features (ICK-r) as well as the definition of commonly used neural network (NN) architectures and kernel functions.
  - `bnn.py`: Contains Bayesian NN-implied kernels whose architecture can be customized
  - `kernel_fn.py`: Contains commonly used kernel functions for both Nystrom approximation and Random Fourier Features implementation
  - `nn.py`: Contains NN-implied kernels whose architecture can be customized
  - `nystrom.py`: Contains user-specified implicit kernels based on Nystrom approximation
  - `rff.py`: Contains user-specified implicit kernels based on Random Fourier Features (working in progress)
- `model/ick.py`: This module contains the implementation of the ICK framework given the implicit kernels.
- `utils`: This module contains the utility and helper functions used for fitting the ICK model.
  - `constants.py`: Contains constant variables used for other scripts in this directory
  - `data_generator.py`: Contains the definition of data loaders for generating multimodal data
  - `helpers.py`: Contains helper functions (e.g. train-validation-test split)
  - `trainer.py`: Contains a trainer class for fitting ICK and making predictions

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
