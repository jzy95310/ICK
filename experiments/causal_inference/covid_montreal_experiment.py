import sys, os, random
sys.path.insert(0, '../../')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import trange
from sklearn.preprocessing import StandardScaler

import torch
from torchvision.transforms import Compose, ToTensor, Resize
from kernels.nn import ImplicitConvNet2DKernel
from model.ick import ICK
from model.cmick import CMICK
from utils.train import CMICKEnsembleTrainer
from utils.helpers import *

# To make this notebook's output stable across runs
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 1. Load and preprocess the images and demographic infomation
def load_and_preprocess_data(train_ratio, test_ratio, random_state, include_images=True, demo_features=None):
    def process_img(img, resize=(224,224), mode='L'):
        # Convert the image to black-white and resize
        assert isinstance(resize, tuple) and len(resize) == 2
        transforms = Compose([ToTensor(), Resize(resize)])
        img = transforms(img.convert(mode))
        return np.array(img)
    
    demo_info_dir = '../../data/Covid_Montreal/metadata_PO.csv'
    imgs_dir = '../../data/Covid_Montreal/images/'
    col_names = ['offset','sex','age','RT_PCR_positive','survival','intubated','intubation_present', 
                 'went_icu', 'in_icu']
    cols_to_normalize = ['offset','age']
    if demo_features is None:
        demo_features = col_names
    else:
        assert set(demo_features).issubset(set(col_names))
    demo_info = pd.read_csv(demo_info_dir)
    for c in demo_info.columns:
        if c in cols_to_normalize:
            scaler = StandardScaler()
            demo_info[c] = scaler.fit_transform(demo_info[c].to_numpy().reshape(-1,1)).reshape(-1)
    demo_info['Y'] = demo_info.apply(lambda row: row['Y0'] if row['treatment'] == 0 else row['Y1'], axis=1)

    N = len(demo_info)
    demo_info_train = demo_info.sample(n=int(train_ratio*N), random_state=random_state)
    demo_info = demo_info.drop(demo_info_train.index)
    demo_info_test = demo_info.sample(n=int(test_ratio*N), random_state=random_state)
    demo_info_val = demo_info.drop(demo_info_test.index)
    D_train, D_val, D_test = np.array(demo_info_train[demo_features]), np.array(demo_info_val[demo_features]), np.array(demo_info_test[demo_features])
    X_train = np.array([process_img(x) for x in [Image.open(imgs_dir+list(demo_info_train['filename'])[i]) for i in range(len(demo_info_train))]])
    X_val = np.array([process_img(x) for x in [Image.open(imgs_dir+list(demo_info_train['filename'])[i]) for i in range(len(demo_info_val))]])
    X_test = np.array([process_img(x) for x in [Image.open(imgs_dir+list(demo_info_train['filename'])[i]) for i in range(len(demo_info_test))]])
    T_train, T_val, T_test = np.array(demo_info_train[['treatment']]), np.array(demo_info_val[['treatment']]), np.array(demo_info_test[['treatment']])
    Y_train, Y_val, Y_test = np.array(demo_info_train[['Y']]), np.array(demo_info_val[['Y']]), np.array(demo_info_test[['Y']])
    Y0_train, Y0_val, Y0_test = np.array(demo_info_train[['Y0']]), np.array(demo_info_val[['Y0']]), np.array(demo_info_test[['Y0']])
    Y1_train, Y1_val, Y1_test = np.array(demo_info_train[['Y1']]), np.array(demo_info_val[['Y1']]), np.array(demo_info_test[['Y1']])
    
    data_train, data_val, data_test = [T_train], [T_val], [T_test]
    if include_images:
        data_train.append(X_train)
        data_val.append(X_val)
        data_test.append(X_test)
    if len(demo_features) > 0:
        data_train.append(D_train)
        data_val.append(D_val)
        data_test.append(D_test)
    data = {'X_train': X_train, 'T_train': T_train, 'D_train': D_train, 'Y_train': Y_train, 'Y0_train': Y0_train, 'Y1_train': Y1_train, 
            'X_val': X_val, 'T_val': T_val, 'D_val': D_val, 'Y_val': Y_val, 'Y0_val': Y0_val, 'Y1_val': Y1_val, 
            'X_test': X_test, 'T_test': T_test, 'D_test': D_test, 'Y_test': Y_test, 'Y0_test': Y0_test, 'Y1_test': Y1_test}
    data_generators = create_generators_from_data(
        x_train=data_train, y_train=Y_train, 
        x_val=data_val, y_val=Y_val,
        x_test=data_test, y_test=Y_test, 
        train_batch_size=16, val_batch_size=16, test_batch_size=16
    )
    del X_train, X_val, X_test, D_train, D_val, D_test, T_train, T_val, T_test, Y_train, Y_val, Y_test
    return data_generators, data


# 2. Build, train, and evaluate CMNN model
# 2.1 Image information only
def fit_evaluate_cmnn_ensemble_image_only(input_width, input_height, in_channels, data_generators, 
                                          data, lr, treatment_index=0):
    alpha11, alpha12, alpha13 = 1.0, 1.0, 1.0
    num_estimators = 10
    
    ensemble, ensemble_weights = [], {}
    for i in range(num_estimators):
        f11 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 512, 
                    'num_blocks': 1, 
                    'activation': 'softplus', 
                }
            }
        )
        f12 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 512, 
                    'num_blocks': 1, 
                    'activation': 'softplus', 
                }
            }
        )
        f13 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 512, 
                    'num_blocks': 1, 
                    'activation': 'softplus', 
                }
            }
        )
        baselearner = CMICK(
            control_components=[f11], treatment_components=[f12], shared_components=[f13],
            control_coeffs=[alpha11], treatment_coeffs=[alpha12], shared_coeffs=[alpha13], 
            coeff_trainable=True
        )
        ensemble.append(baselearner)
    
    # The index of "T_train" in "data_train" is 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 10
    trainer = CMICKEnsembleTrainer(
        model=ensemble,
        data_generators=data_generators,
        optim=optim,
        optim_params=optim_params, 
        model_save_dir=None,
        device=device,
        epochs=epochs,
        patience=patience, 
        treatment_index=treatment_index
    )
    trainer.train()
    
    mean_test_pred, std_test_pred, y_test_true = trainer.predict()
    y0_test, y1_test = data['Y0_test'], data['Y1_test']
    pehe_test = np.sqrt(np.mean(((mean_test_pred[:,1] - mean_test_pred[:,0]) - (y1_test - y0_test)) ** 2))
    print('PEHE (CMNN with image only):             %.4f' % (pehe_test))
    
    return pehe_test


# 2.2 Demographic information only
def fit_evaluate_cmnn_ensemble_demo_only(input_dim, data_generators, data, lr, treatment_index=0):
    alpha11, alpha12, alpha13 = 1.0, 1.0, 1.0
    num_estimators = 10
    
    ensemble, ensemble_weights = [], {}
    for i in range(num_estimators):
        f11 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus'
                }
            }
        )
        f12 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus'
                }
            }
        )
        f13 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus'
                }
            }
        )
        baselearner = CMICK(
            control_components=[f11], treatment_components=[f12], shared_components=[f13],
            control_coeffs=[alpha11], treatment_coeffs=[alpha12], shared_coeffs=[alpha13], 
            coeff_trainable=True
        )
        ensemble.append(baselearner)
    
    # The index of "T_train" in "data_train" is 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 10
    trainer = CMICKEnsembleTrainer(
        model=ensemble,
        data_generators=data_generators,
        optim=optim,
        optim_params=optim_params, 
        model_save_dir=None,
        device=device,
        epochs=epochs,
        patience=patience, 
        treatment_index=treatment_index
    )
    trainer.train()
    
    mean_test_pred, std_test_pred, y_test_true = trainer.predict()
    y0_test, y1_test = data['Y0_test'], data['Y1_test']
    pehe_test = np.sqrt(np.mean(((mean_test_pred[:,1] - mean_test_pred[:,0]) - (y1_test - y0_test)) ** 2))
    print('PEHE (CMNN with demographic info only):             %.4f' % (pehe_test))
    
    return pehe_test


# 2.3 Image + demographic information
def fit_evaluate_cmnn_ensemble_image_demo(input_width, input_height, in_channels, demo_dim, data_generators, 
                                          data, lr, treatment_index=0):
    alpha11, alpha12, alpha13 = 1.0, 1.0, 1.0
    num_estimators = 10
    
    ensemble, ensemble_weights = [], {}
    for i in range(num_estimators):
        f11 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel','ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'activation': 'softplus'
                }, 
                'ImplicitDenseNetKernel':{
                    'input_dim': demo_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus'
                }
            }
        )
        f12 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel','ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'activation': 'softplus'
                }, 
                'ImplicitDenseNetKernel':{
                    'input_dim': demo_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus'
                }
            }
        )
        f13 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel','ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'activation': 'softplus'
                }, 
                'ImplicitDenseNetKernel':{
                    'input_dim': demo_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus'
                }
            }
        )
        baselearner = CMICK(
            control_components=[f11], treatment_components=[f12], shared_components=[f13],
            control_coeffs=[alpha11], treatment_coeffs=[alpha12], shared_coeffs=[alpha13], 
            coeff_trainable=True
        )
        ensemble.append(baselearner)
        
    # The index of "T_train" in "data_train" is 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 10
    trainer = CMICKEnsembleTrainer(
        model=ensemble,
        data_generators=data_generators,
        optim=optim,
        optim_params=optim_params, 
        model_save_dir=None,
        device=device,
        epochs=epochs,
        patience=patience, 
        treatment_index=treatment_index
    )
    trainer.train()
    
    mean_test_pred, std_test_pred, y_test_true = trainer.predict()
    y0_test, y1_test = data['Y0_test'], data['Y1_test']
    pehe_test = np.sqrt(np.mean(((mean_test_pred[:,1] - mean_test_pred[:,0]) - (y1_test - y0_test)) ** 2))
    print('PEHE (CMNN with image and demographic info):             %.4f' % (pehe_test))
    
    return pehe_test


# 3. Build, train, and evaluate CMICK model
# 3.1 Image + age information
def fit_evaluate_cmick_ensemble_image_demo(input_width, input_height, in_channels, demo_range, data_generators, 
                                           data, lr, treatment_index=0):
    alpha11, alpha12, alpha13 = 1.0, 1.0, 0.1
    num_estimators = 10
    
    ensemble, ensemble_weights = [], {}
    for i in range(num_estimators):
        f11 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel','ImplicitNystromKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 32, 
                    'num_blocks': 1, 
                    'activation': 'relu'
                }, 
                'ImplicitNystromKernel':{
                    'kernel_func': linear_kernel_nys, 
                    'params': ['std','c','noise'], 
                    'vals': [1.,0.25,0.5], 
                    'trainable': [True,True,True], 
                    'alpha': 1e-5, 
                    'num_inducing_points': 32, 
                    'nys_space': demo_range
                }
            }
        )
        f12 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel','ImplicitNystromKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 32, 
                    'num_blocks': 1, 
                    'activation': 'softplus'
                }, 
                'ImplicitNystromKernel':{
                    'kernel_func': linear_kernel_nys, 
                    'params': ['std','c','noise'], 
                    'vals': [1.,0.25,0.5], 
                    'trainable': [True,True,True], 
                    'alpha': 1e-5, 
                    'num_inducing_points': 32, 
                    'nys_space': demo_range
                }
            }
        )
        f13 = ICK(
            kernel_assignment=['ImplicitConvNet2DKernel','ImplicitNystromKernel'],
            kernel_params={
                'ImplicitConvNet2DKernel':{
                    'input_width': input_width,
                    'input_height': input_height, 
                    'in_channels': in_channels,
                    'num_intermediate_channels': 64, 
                    'latent_feature_dim': 32, 
                    'num_blocks': 1, 
                    'activation': 'softplus'
                }, 
                'ImplicitNystromKernel':{
                    'kernel_func': linear_kernel_nys, 
                    'params': ['std','c','noise'], 
                    'vals': [1.,0.25,0.5], 
                    'trainable': [True,True,True], 
                    'alpha': 1e-5, 
                    'num_inducing_points': 32, 
                    'nys_space': demo_range
                }
            }
        )
        baselearner = CMICK(
            control_components=[f11], treatment_components=[f12], shared_components=[f13],
            control_coeffs=[alpha11], treatment_coeffs=[alpha12], shared_coeffs=[alpha13], 
            coeff_trainable=True
        )
        ensemble.append(baselearner)
    
    # The index of "T_train" in "data_train" is 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 10
    trainer = CMICKEnsembleTrainer(
        model=ensemble,
        data_generators=data_generators,
        optim=optim,
        optim_params=optim_params, 
        model_save_dir=None,
        device=device,
        epochs=epochs,
        patience=patience, 
        treatment_index=treatment_index
    )
    trainer.train()
    
    mean_test_pred, std_test_pred, y_test_true = trainer.predict()
    y0_test, y1_test = data['Y0_test'], data['Y1_test']
    pehe_test = np.sqrt(np.mean(((mean_test_pred[:,1] - mean_test_pred[:,0]) - (y1_test - y0_test)) ** 2))
    print('PEHE (CMICK with image and demographic info):             %.4f' % (pehe_test))
    
    return pehe_test


# 4. Benchmark 1: CFRNet (with image only)
def fit_and_evaluate_cfrnet(input_width, input_height, in_channels, phi_depth, phi_width, h_depth, h_width, 
                            data_generators, data, lr, alpha, metric='W2', treatment_index=0):
    cfrnet = Conv2DCFRNet(input_width, input_height, in_channels, phi_depth, phi_width, h_depth, h_width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 10
    trainer = CFRNetTrainer(
        model=cfrnet,
        data_generators=data_generators,
        optim=optim,
        optim_params=optim_params, 
        model_save_dir=None,
        loss_fn=CFRLoss(alpha=alpha,metric=metric),
        device=device,
        epochs=epochs,
        patience=patience, 
        treatment_index=treatment_index
    )
    trainer.train()
    
    y_test_pred, y_test_true = trainer.predict()
    y0_test, y1_test = data['Y0_test'], data['Y1_test']
    pehe_test = np.sqrt(np.mean(((y_test_pred[:,1] - y_test_pred[:,0]) - (y1_test - y0_test)) ** 2))
    print('PEHE (CFRNet):             %.4f' % (pehe_test))
    
    return pehe_test


# 5. Benchmark 2: DCN-PD
def fit_and_evaluate_dcn_pd(input_width, input_height, in_channels, shared_conv_blocks, shared_channels, 
                           idiosyncratic_depth, idiosyncratic_width, data_generators, data, lr, treatment_index=0):
    dcn_pd = Conv2DDCNPD(
        input_width=input_width, 
        input_height=input_height, 
        in_channels=in_channels, 
        shared_conv_blocks=shared_conv_blocks, 
        shared_channels=shared_channels, 
        idiosyncratic_depth=idiosyncratic_depth, 
        idiosyncratic_width=idiosyncratic_width,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 10
    trainer = DCNTrainer(
        model=dcn_pd,
        data_generators=data_generators,
        optim=optim,
        optim_params=optim_params, 
        model_save_dir=None,
        device=device,
        epochs=epochs,
        patience=patience, 
        treatment_index=treatment_index
    )
    trainer.train()
    
    y_test_pred, y_test_true = trainer.predict()
    y0_test, y1_test = data['Y0_test'], data['Y1_test']
    pehe_test = np.sqrt(np.mean(((y_test_pred[:,1] - y_test_pred[:,0]) - (y1_test - y0_test)) ** 2))
    print('PEHE (DCN-PD):             %.4f' % (pehe_test))
    
    return pehe_test


# Main function
def main():
    train_ratio, test_ratio = 0.60, 0.20
    
    # CMNN
    data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=1, demo_features=[])
    input_width, input_height, in_channels = data['X_train'].shape[2], data['X_train'].shape[3], data['X_train'].shape[1]
    sqrt_pehe_cmnn_image_only = fit_evaluate_cmnn_ensemble_image_only(input_width, input_height, in_channels, 
                                                                      data_generators, data, lr=5e-5)
    data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=1, include_images=False)
    demo_input_dim = data['D_train'].shape[1]
    sqrt_pehe_cmnn_demo_only = fit_evaluate_cmnn_ensemble_demo_only(demo_input_dim, data_generators, data, lr=5e-4)
    data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=1)
    demo_input_dim = data['D_train'].shape[1]
    sqrt_pehe_cmnn_image_demo = fit_evaluate_cmnn_ensemble_image_demo(input_width, input_height, in_channels, 
                                                                      demo_input_dim, data_generators, data, lr=1e-4)
    
    # CMICK
    data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=1, demo_features=['age'])
    input_width, input_height, in_channels = data['X_train'].shape[2], data['X_train'].shape[3], data['X_train'].shape[1]
    demo_range = [[np.min(data['D_train'][:,d]), np.max(data['D_train'][:,d])] for d in range(data['D_train'].shape[1])]
    sqrt_pehe_cmick_image_demo = fit_evaluate_cmick_ensemble_image_demo(input_width, input_height, in_channels, 
                                                                        demo_range, data_generators, data, lr=1e-4)
    
    # Benchmarks
    data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=1, demo_features=[])
    input_width, input_height, in_channels = data['X_train'].shape[2], data['X_train'].shape[3], data['X_train'].shape[1]
    sqrt_pehe_cfrnet_wass = fit_and_evaluate_cfrnet(
        input_width, input_height, in_channels, 2, 512, 2, 512, data_generators, data, lr=1e-5,
        alpha=1e-2, metric='W2'
    )
    sqrt_pehe_dcn_pd = fit_and_evaluate_dcn_pd(
        input_width, input_height, in_channels, 2, 64, 2, 512, data_generators, data, lr=1e-5
    )
    
    print('PEHE (CMNN with image only):             %.4f' % (sqrt_pehe_cmnn_image_only))
    print('PEHE (CMNN with demographic info only):             %.4f' % (sqrt_pehe_cmnn_demo_only))
    print('PEHE (CMNN with image and demographic info):             %.4f' % (sqrt_pehe_cmnn_image_demo))
    print('PEHE (CMICK with image and demographic info):             %.4f' % (sqrt_pehe_cmick_image_demo))
    print('PEHE (CFRNet):             %.4f' % (sqrt_pehe_cfrnet_wass))
    print('PEHE (DCN-PD):             %.4f' % (sqrt_pehe_dcn_pd))

if __name__ == "__main__":
    main()

