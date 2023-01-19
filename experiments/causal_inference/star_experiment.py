import sys, os, random
sys.path.insert(0, '../../')
import os  
import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
from tqdm.notebook import trange
from sklearn.preprocessing import StandardScaler

import torch
from torchvision.transforms import Compose, ToTensor, Resize
from kernels.nn import ImplicitConvNet2DKernel
from kernels.kernel_fn import linear_kernel_nys, sq_exp_kernel_nys
from model.ick import ICK
from model.cmick import CMICK, AdditiveCMICK
from benchmarks.cfrnet import DenseCFRNet, Conv2DCFRNet
from benchmarks.dcn_pd import DenseDCNPD, Conv2DDCNPD
from benchmarks.donut import DenseDONUT, Conv2DDONUT
from benchmarks.train_benchmarks import CFRNetTrainer, DCNTrainer, DONUTTrainer
from utils.train import CMICKEnsembleTrainer
from utils.metrics import policy_risk
from utils.losses import *
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
def load_and_preprocess_data(train_ratio, test_ratio, random_state, include_images=True, demo_features=None, batch_size=8):
    def process_img(img, resize=(100,100), mode='RGB'):
        assert isinstance(resize, tuple) and len(resize) == 2
        transforms = Compose([ToTensor(), Resize(resize)])
        img = transforms(img.convert(mode))
        return np.array(img)
    
    demo_info_dir = '../../data/STAR/STAR.csv'
    imgs_dir = '../../data/STAR/face_pic/'
    col_names = ['trace.1', 'trace.2',
                 'degree.2', 'degree.3', 'degree.5', 'degree.6',
                 'careerladder.1', 'careerladder.2', 'careerladder.3', 'careerladder.4', 'careerladder.5', 'careerladder.6',
                 'teachingexperience']
    if demo_features is None:
        demo_features = col_names
    else:
        assert set(demo_features).issubset(set(col_names))
    demo_info = pd.read_csv(demo_info_dir)
    ate = demo_info['ATE'][0]   # ATE is the same for all rows in demo_info

    N = len(demo_info)
    demo_info_train = demo_info.sample(n=int(train_ratio*N), random_state=random_state)
    demo_info = demo_info.drop(demo_info_train.index)
    demo_info_test = demo_info.sample(n=int(test_ratio*N), random_state=random_state)
    demo_info_val = demo_info.drop(demo_info_test.index)
    D_train, D_val, D_test = np.array(demo_info_train[demo_features]), np.array(demo_info_val[demo_features]), np.array(demo_info_test[demo_features])

    X_train = np.array([process_img(x) for x in [Image.open(imgs_dir+list(demo_info_train['image_file'])[i]) for i in range(len(demo_info_train))]])
    X_val = np.array([process_img(x) for x in [Image.open(imgs_dir+list(demo_info_train['image_file'])[i]) for i in range(len(demo_info_val))]])
    X_test = np.array([process_img(x) for x in [Image.open(imgs_dir+list(demo_info_train['image_file'])[i]) for i in range(len(demo_info_test))]])
    
    T_train, T_val, T_test = np.array(demo_info_train[['treatment']]), np.array(demo_info_val[['treatment']]), np.array(demo_info_test[['treatment']])
    Y_train, Y_val, Y_test = np.array(demo_info_train[['score']]), np.array(demo_info_val[['score']]), np.array(demo_info_test[['score']])
    
    data_train, data_val, data_test = [T_train], [T_val], [T_test]
    if include_images:
        data_train.append(X_train)
        data_val.append(X_val)
        data_test.append(X_test)
    if len(demo_features) > 0:
        data_train.append(D_train)
        data_val.append(D_val)
        data_test.append(D_test)

    data = {'X_train': X_train, 'T_train': T_train, 'D_train': D_train, 'Y_train': Y_train.squeeze(),
            'X_val': X_val, 'T_val': T_val, 'D_val': D_val, 'Y_val': Y_val.squeeze(), 
            'X_test': X_test, 'T_test': T_test, 'D_test': D_test, 'Y_test': Y_test.squeeze(), 'ATE': ate}
    data_generators = create_generators_from_data(
        x_train=data_train, y_train=Y_train, 
        x_val=data_val, y_val=Y_val,
        x_test=data_test, y_test=Y_test, 
        train_batch_size=batch_size, val_batch_size=batch_size, test_batch_size=batch_size, 
        drop_last=False
    )
    del X_train, X_val, X_test, D_train, D_val, D_test, T_train, T_val, T_test, Y_train, Y_val, Y_test
    return data_generators, data


# 2. Build, train, and evaluate CMNN model
# 2.1 Image information only
def fit_evaluate_cmnn_ensemble_image_only(input_width, input_height, in_channels, data_generators, 
                                          data, lr, treatment_index=0, load_weights=False):
    alpha11, alpha12, alpha13 = 1.0, 1.0, 0.1
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
                    'skip_connection': True
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
                    'skip_connection': True
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
                    'skip_connection': True
                }
            }
        )
        baselearner = CMICK(
            control_components=[f11], treatment_components=[f12], shared_components=[f13],
            control_coeffs=[alpha11], treatment_coeffs=[alpha12], shared_coeffs=[alpha13], 
            coeff_trainable=True
        )
        if load_weights:
            baselearner.load_state_dict(torch.load('./checkpoints/cmde_img_star.pt')['model_'+str(i+1)])
        else:
            ensemble_weights['model_'+str(i+1)] = baselearner.state_dict()
        ensemble.append(baselearner)

    if not load_weights:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
            torch.save(ensemble_weights, './checkpoints/cmde_img_star.pt')
    
    # The index of "T_train" in "data_train" is 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(mean_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(mean_test_pred[:,1] - mean_test_pred[:,0]))

    print('Policy risk (CMDE with image only):             %.4f' % (r_pol))
    print('ATE error (CMDE with image only):             %.4f' % (eps_ate))
    
    return r_pol, eps_ate


# 2.2 Demographic information only
def fit_evaluate_cmnn_ensemble_demo_only(input_dim, data_generators, data, lr, treatment_index=0, 
                                         load_weights=False):
    alpha11, alpha12, alpha13 = 1.0, 1.0, 0.1
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
                    'activation': 'softplus', 
                    'skip_connection': True
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
                    'activation': 'softplus', 
                    'skip_connection': True
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
                    'activation': 'softplus', 
                    'skip_connection': True
                }
            }
        )
        baselearner = CMICK(
            control_components=[f11], treatment_components=[f12], shared_components=[f13],
            control_coeffs=[alpha11], treatment_coeffs=[alpha12], shared_coeffs=[alpha13], 
            coeff_trainable=True
        )
        if load_weights:
            baselearner.load_state_dict(torch.load('./checkpoints/cmde_demo_star.pt')['model_'+str(i+1)])
        else:
            ensemble_weights['model_'+str(i+1)] = baselearner.state_dict()
        ensemble.append(baselearner)

    if not load_weights:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(ensemble_weights, './checkpoints/cmde_demo_star.pt')
    
    # The index of "T_train" in "data_train" is 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(mean_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(mean_test_pred[:,1] - mean_test_pred[:,0]))
    
    print('Policy risk (CMDE with demographic info only):             %.4f' % (r_pol))
    print('ATE error (CMDE with demographic info only):             %.4f' % (eps_ate))
    
    return r_pol, eps_ate


# 2.3 Image + demographic information
def fit_evaluate_cmnn_ensemble_image_demo(input_width, input_height, in_channels, demo_dim, data_generators, 
                                          data, lr, treatment_index=0, load_weights=False):
    alpha11, alpha12, alpha13 = 1.0, 1.0, 0.1
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
                    'activation': 'softplus', 
                    'skip_connection': True
                }, 
                'ImplicitDenseNetKernel':{
                    'input_dim': demo_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus', 
                    'skip_connection': True
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
                    'activation': 'softplus', 
                    'skip_connection': True
                }, 
                'ImplicitDenseNetKernel':{
                    'input_dim': demo_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus', 
                    'skip_connection': True
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
                    'activation': 'softplus', 
                    'skip_connection': True
                }, 
                'ImplicitDenseNetKernel':{
                    'input_dim': demo_dim, 
                    'latent_feature_dim': 128, 
                    'num_blocks': 1, 
                    'num_layers_per_block': 1,
                    'num_units': 512,
                    'activation': 'softplus', 
                    'skip_connection': True
                }
            }
        )
        baselearner = CMICK(
            control_components=[f11], treatment_components=[f12], shared_components=[f13],
            control_coeffs=[alpha11], treatment_coeffs=[alpha12], shared_coeffs=[alpha13], 
            coeff_trainable=True
        )
        if load_weights:
            baselearner.load_state_dict(torch.load('./checkpoints/cmde_img_demo_star.pt')['model_'+str(i+1)])
        else:
            ensemble_weights['model_'+str(i+1)] = baselearner.state_dict()
        ensemble.append(baselearner)

    if not load_weights:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(ensemble_weights, './checkpoints/cmde_img_demo_star.pt')
        
    # The index of "T_train" in "data_train" is 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(mean_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(mean_test_pred[:,1] - mean_test_pred[:,0]))
    
    print('Policy risk (CMDE with image and demographic info):             %.4f' % (r_pol))
    print('ATE error (CMDE with image and demographic info):             %.4f' % (eps_ate))
    
    return r_pol, eps_ate


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
                    'activation': 'softplus', 
                    'skip_connection': True
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
                    'activation': 'softplus', 
                    'skip_connection': True
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
                    'activation': 'softplus', 
                    'skip_connection': True
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
    epochs, patience = 1000, 15
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(mean_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(mean_test_pred[:,1] - mean_test_pred[:,0]))
    
    print('Policy risk (CMICK with image and demographic info):             %.4f' % (r_pol))
    print('ATE error (CMICK with image and demographic info):             %.4f' % (eps_ate))
    
    return r_pol, eps_ate


# 4. Benchmark 1: CFRNet
# 4.1 Image only
def fit_evaluate_cfrnet_image_only(input_width, input_height, in_channels, phi_depth, phi_width, h_depth, h_width, 
                                   data_generators, data, lr, alpha, metric='W2', treatment_index=0, 
                                   load_weights=False):
    cfrnet = Conv2DCFRNet(input_width, input_height, in_channels, phi_depth, phi_width, h_depth, h_width,
                          activation='softplus', skip_connection=True)
    if load_weights:
        cfrnet.load_state_dict(torch.load('./checkpoints/cfrnet_img_star.pt'))
    else:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(cfrnet.state_dict(), './checkpoints/cfrnet_img_star.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(y_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(y_test_pred[:,1] - y_test_pred[:,0]))
    
    print('Policy risk (CFRNet-Wass with image only):             %.4f' % (r_pol))
    print('ATE error (CFRNet-Wass with image only):             %.4f' % (eps_ate))

    return r_pol, eps_ate


# 4.2 Demographic information only
def fit_evaluate_cfrnet_demo_only(input_dim, phi_depth, phi_width, h_depth, h_width, data_generators, 
                                  data, lr, alpha, metric='W2', treatment_index=0, load_weights=False):
    cfrnet = DenseCFRNet(input_dim, phi_depth, phi_width, h_depth, h_width, activation='softplus', 
                         skip_connection=True)
    if load_weights:
        cfrnet.load_state_dict(torch.load('./checkpoints/cfrnet_demo_star.pt'))
    else:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(cfrnet.state_dict(), './checkpoints/cfrnet_demo_star.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(y_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(y_test_pred[:,1] - y_test_pred[:,0]))
    
    print('Policy risk (CFRNet-Wass with demo only):             %.4f' % (r_pol))
    print('ATE error (CFRNet-Wass with demo only):             %.4f' % (eps_ate))

    return r_pol, eps_ate


# 5. Benchmark 2: DCN-PD
# 5.1 Image only
def fit_evaluate_dcn_pd_image_only(input_width, input_height, in_channels, shared_conv_blocks, shared_channels, 
                                   idiosyncratic_depth, idiosyncratic_width, data_generators, data, lr, 
                                   treatment_index=0, load_weights=False):
    dcn_pd = Conv2DDCNPD(
        input_width=input_width, 
        input_height=input_height, 
        in_channels=in_channels, 
        shared_conv_blocks=shared_conv_blocks, 
        shared_channels=shared_channels, 
        idiosyncratic_depth=idiosyncratic_depth, 
        idiosyncratic_width=idiosyncratic_width,
        activation='softplus', 
        skip_connection=True
    )
    if load_weights:
        dcn_pd.load_state_dict(torch.load('./checkpoints/dcn_pd_img_star.pt'))
    else:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(dcn_pd.state_dict(), './checkpoints/dcn_pd_img_star.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(y_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(y_test_pred[:,1] - y_test_pred[:,0]))
    
    print('Policy risk (DCN-PD with image only):             %.4f' % (r_pol))
    print('ATE error (DCN-PD with image only):             %.4f' % (eps_ate))
    
    return r_pol, eps_ate


# 5.2 Demographic information only
def fit_evaluate_dcn_pd_demo_only(input_dim, shared_depth, shared_width, idiosyncratic_depth, idiosyncratic_width,
                                  data_generators, data, lr, treatment_index=0, load_weights=False):
    dcn_pd = DenseDCNPD(
        input_dim=input_dim, 
        shared_depth=shared_depth, 
        shared_width=shared_width, 
        idiosyncratic_depth=idiosyncratic_depth, 
        idiosyncratic_width=idiosyncratic_width, 
        activation='softplus', 
        skip_connection=True
    )
    if load_weights:
        dcn_pd.load_state_dict(torch.load('./checkpoints/dcn_pd_demo_star.pt'))
    else:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(dcn_pd.state_dict(), './checkpoints/dcn_pd_demo_star.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(y_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(y_test_pred[:,1] - y_test_pred[:,0]))
    
    print('Policy risk (DCN-PD with demo only):             %.4f' % (r_pol))
    print('ATE error (DCN-PD with demo only):             %.4f' % (eps_ate))

    return r_pol, eps_ate


# 6. Benchmark 3: DONUT
# 6.1 Image only
def fit_evaluate_donut_image_only(input_width, input_height, in_channels, phi_depth, phi_width, h_depth, h_width, 
                                  data_generators, data, lr, treatment_index=0, load_weights=False):
    donut = Conv2DDONUT(input_width, input_height, in_channels, phi_depth, phi_width, h_depth, h_width,
                        activation='softplus', skip_connection=True)
    if load_weights:
        donut.load_state_dict(torch.load('./checkpoints/donut_img_star.pt'))
    else:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(donut.state_dict(), './checkpoints/donut_img_star.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
    trainer = DONUTTrainer(
        model=donut,
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(y_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(y_test_pred[:,1] - y_test_pred[:,0]))
    
    print('Policy risk (DONUT with image only):             %.4f' % (r_pol))
    print('ATE error (DONUT with image only):             %.4f' % (eps_ate))
    
    return r_pol, eps_ate


# 6.2 Demographic information only
def fit_evaluate_donut_demo_only(input_dim, phi_depth, phi_width, h_depth, h_width, data_generators, 
                                 data, lr, treatment_index=0, load_weights=False):
    donut = DenseDONUT(input_dim, phi_depth, phi_width, h_depth, h_width, activation='softplus', 
                       skip_connection=True)
    if load_weights:
        donut.load_state_dict(torch.load('./checkpoints/donut_demo_star.pt'))
    else:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(donut.state_dict(), './checkpoints/donut_demo_star.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 15
    trainer = DONUTTrainer(
        model=donut,
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
    y_test, t_test = data['Y_test'], data['T_test']
    r_pol = policy_risk(y_test_pred, y_test, t_test)
    eps_ate = np.abs(data['ATE'] - np.mean(y_test_pred[:,1] - y_test_pred[:,0]))
    
    print('Policy risk (DONUT with demo only):             %.4f' % (r_pol))
    print('ATE error (DONUT with demo only):             %.4f' % (eps_ate))
    
    return r_pol, eps_ate


def main():
    train_ratio, test_ratio = 0.4, 0.4
    res = {'R_pol': defaultdict(list), 'eps_ate': defaultdict(list)}
    num_repetitions = 10
    load_weight = False
    
    for rep in range(num_repetitions):
        print("Repetition: " + str(rep))

        # CMDE
        # CMDE with image only
        data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=rep, demo_features=[])
        input_width, input_height, in_channels = data['X_train'].shape[2], data['X_train'].shape[3], data['X_train'].shape[1]
        r_pol_cmde_image_only, eps_ate_cmde_image_only = fit_evaluate_cmnn_ensemble_image_only(input_width, input_height, in_channels, 
                                                                       data_generators, data, lr=5e-5, load_weights=load_weight)
        # CMDE with demo only
        data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=rep, include_images=False)
        demo_input_dim = data['D_train'].shape[1]
        r_pol_cmde_demo_only, eps_ate_cmde_demo_only = fit_evaluate_cmnn_ensemble_demo_only(demo_input_dim, data_generators, data, lr=5e-4, 
                                                                            load_weights=load_weight)
        # CMDE with image and demo
        data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=rep)
        demo_input_dim = data['D_train'].shape[1]
        r_pol_cmde_image_demo, eps_ate_cmde_image_demo = fit_evaluate_cmnn_ensemble_image_demo(input_width, input_height, in_channels, 
                                                                        demo_input_dim, data_generators, data, 
                                                                        lr=1e-4, load_weights=load_weight)

#       # CMICK
#       data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=rep, demo_features=['age'])
#       input_width, input_height, in_channels = data['X_train'].shape[2], data['X_train'].shape[3], data['X_train'].shape[1]
#       demo_range = [[np.min(data['D_train'][:,d]), np.max(data['D_train'][:,d])] for d in range(data['D_train'].shape[1])]
#       sqrt_pehe_cmick_image_demo = fit_evaluate_cmick_ensemble_image_demo(input_width, input_height, in_channels, 
#                                                                             demo_range, data_generators, data, lr=1e-4)

        # Benchmarks
        # With image only
        data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=rep, demo_features=[])
        input_width, input_height, in_channels = data['X_train'].shape[2], data['X_train'].shape[3], data['X_train'].shape[1]
        r_pol_cfrnet_wass_image_only, eps_ate_cfrnet_wass_image_only = fit_evaluate_cfrnet_image_only(
            input_width, input_height, in_channels, 2, 512, 2, 512, data_generators, data, lr=1e-5,
            alpha=1e-2, metric='W2', load_weights=load_weight
        )
        r_pol_dcn_pd_image_only, eps_ate_dcn_pd_image_only = fit_evaluate_dcn_pd_image_only(
            input_width, input_height, in_channels, 2, 64, 2, 512, data_generators, data, lr=1e-5, 
            load_weights=load_weight
        )
        r_pol_donut_image_only, eps_ate_donut_image_only = fit_evaluate_donut_image_only(
            input_width, input_height, in_channels, 2, 512, 2, 512, data_generators, data, lr=1e-5, 
            load_weights=load_weight
        )
        
        # With demographic information only
        data_generators, data = load_and_preprocess_data(train_ratio, test_ratio, random_state=rep, include_images=False)
        demo_input_dim = data['D_train'].shape[1]
        r_pol_cfrnet_wass_demo_only, eps_ate_cfrnet_wass_demo_only = fit_evaluate_cfrnet_demo_only(
            demo_input_dim, 2, 512, 2, 512, data_generators, data, lr=1e-5, alpha=1e-2, metric='W2', 
            load_weights=load_weight
        )
        r_pol_dcn_pd_demo_only, eps_ate_dcn_pd_demo_only = fit_evaluate_dcn_pd_demo_only(
            demo_input_dim, 2, 512, 2, 512, data_generators, data, lr=1e-5, load_weights=load_weight
        )
        r_pol_donut_demo_only, eps_ate_donut_demo_only = fit_evaluate_donut_demo_only(
            demo_input_dim, 2, 512, 2, 512, data_generators, data, lr=1e-5, load_weights=load_weight
        )
            
        print('Policy risk (CMDE with image only):             %.4f' % (r_pol_cmde_image_only))
        print('Policy risk (CMDE with demographic info only):             %.4f' % (r_pol_cmde_demo_only))
        print('Policy risk (CMDE with image and demographic info):             %.4f' % (r_pol_cmde_image_demo))
#         print('Policy risk (CMICK with image and demographic info):             %.4f' % (r_pol_cmick_image_demo))
        print('Policy risk (CFRNet-Wass with image only):             %.4f' % (r_pol_cfrnet_wass_image_only))
        print('Policy risk (CFRNet-Wass with demo only):             %.4f' % (r_pol_cfrnet_wass_demo_only))
        print('Policy risk (DCN-PD with image only):             %.4f' % (r_pol_dcn_pd_image_only))
        print('Policy risk (DCN-PD with demo only):             %.4f' % (r_pol_dcn_pd_demo_only))
        print('Policy risk (DONUT with image only):             %.4f' % (r_pol_donut_image_only))
        print('Policy risk (DONUT with demo only):             %.4f' % (r_pol_donut_demo_only))

        print('ATE error (CMDE with image only):             %.4f' % (eps_ate_cmde_image_only))
        print('ATE error (CMDE with demographic info only):             %.4f' % (eps_ate_cmde_demo_only))
        print('ATE error (CMDE with image and demographic info):             %.4f' % (eps_ate_cmde_image_demo))
#         print('ATE error (CMICK with image and demographic info):             %.4f' % (eps_ate_cmick_image_demo))
        print('ATE error (CFRNet-Wass with image only):             %.4f' % (eps_ate_cfrnet_wass_image_only))
        print('ATE error (CFRNet-Wass with demo only):             %.4f' % (eps_ate_cfrnet_wass_demo_only))
        print('ATE error (DCN-PD with image only):             %.4f' % (eps_ate_dcn_pd_image_only))
        print('ATE error (DCN-PD with demo only):             %.4f' % (eps_ate_dcn_pd_demo_only))
        print('ATE error (DONUT with image only):             %.4f' % (eps_ate_donut_image_only))
        print('ATE error (DONUT with demo only):             %.4f' % (eps_ate_donut_demo_only))

        res['R_pol']['cmde_image_only'].append(r_pol_cmde_image_only)
        res['R_pol']['cmde_demo_only'].append(r_pol_cmde_demo_only)
        res['R_pol']['cmde_image_demo'].append(r_pol_cmde_image_demo)
        res['R_pol']['cfrnet_wass_image_only'].append(r_pol_cfrnet_wass_image_only)
        res['R_pol']['cfrnet_wass_demo_only'].append(r_pol_cfrnet_wass_demo_only)
        res['R_pol']['dcn_pd_image_only'].append(r_pol_dcn_pd_image_only)
        res['R_pol']['dcn_pd_demo_only'].append(r_pol_dcn_pd_demo_only)
        res['R_pol']['donut_image_only'].append(r_pol_donut_image_only)
        res['R_pol']['donut_demo_only'].append(r_pol_donut_demo_only)

        res['eps_ate']['cmde_image_only'].append(eps_ate_cmde_image_only)
        res['eps_ate']['cmde_demo_only'].append(eps_ate_cmde_demo_only)
        res['eps_ate']['cmde_image_demo'].append(eps_ate_cmde_image_demo)
        res['eps_ate']['cfrnet_wass_image_only'].append(eps_ate_cfrnet_wass_image_only)
        res['eps_ate']['cfrnet_wass_demo_only'].append(eps_ate_cfrnet_wass_demo_only)
        res['eps_ate']['dcn_pd_image_only'].append(eps_ate_dcn_pd_image_only)
        res['eps_ate']['dcn_pd_demo_only'].append(eps_ate_dcn_pd_demo_only)
        res['eps_ate']['donut_image_only'].append(eps_ate_donut_image_only)
        res['eps_ate']['donut_demo_only'].append(eps_ate_donut_demo_only)
    
    if not os.path.exists('./results'):
        os.makedirs('./results')
        
    with open('./results/star_results.pkl', 'wb') as fp:
        pkl.dump(res, fp)     
    
    fig, axs = plt.subplots(2, 2, figsize=(15, 14))
    model_names = ['cmde_image_only', 'cfrnet_wass_image_only', 'dcn_pd_image_only', 'donut_image_only', 'cmde_image_demo']
    colors = ['#6671FF', '#FF9997', '#C097FF', '#FA97FF', '#97FFAA']
    labels = ['CMDE-\nimg', 'CFRNet-\nimg', 'DCN-PD-\nimg', 'DONUT-\nimg', 'CMDE-\nmul']
    boxplot_data1 = []
    for i in range(len(model_names)):
        boxplot_data1.append(res['R_pol'][model_names[i]])
    bplot1 = axs[0,0].boxplot(boxplot_data1,
                       sym='', vert=True, 
                       patch_artist=True, labels = labels)
    for patch, color in zip(bplot1['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot1['medians']:
        median.set_color('black')
    axs[0,0].set_ylabel('Policy Risk', fontsize=15)
    axs[0,0].tick_params(axis='both', which='major', labelsize=15)
    axs[0,0].set_facecolor('#F2F2F2')
    axs[0,0].grid(color='white')
    axs[0,0].set_title('Policy risk with only image as input for benchmark models', fontsize=15)

    model_names = ['cmde_demo_only', 'cfrnet_wass_demo_only', 'donut_demo_only', 'cmde_image_demo']
    colors = ['#6671FF', '#FF9997', '#FA97FF', '#97FFAA']
    labels = ['CMDE-\ndem', 'CFRNet-\ndem', 'DONUT-\ndem', 'CMDE-\nmul']
    boxplot_data2 = []
    for i in range(len(model_names)):
        boxplot_data2.append(res['R_pol'][model_names[i]])
    bplot2 = axs[0,1].boxplot(boxplot_data2,
                       sym='', vert=True,
                       patch_artist=True, labels = labels)
    for patch, color in zip(bplot2['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot2['medians']:
        median.set_color('black')
    axs[0,1].set_ylabel('Policy Risk', fontsize=15)
    axs[0,1].tick_params(axis='both', which='major', labelsize=15)
    axs[0,1].set_facecolor('#F2F2F2')
    axs[0,1].grid(color='white')
    axs[0,1].set_title('Policy risk with only demographic information as input\n for benchmark models', fontsize=15)

    model_names = ['cmde_image_only', 'cfrnet_wass_image_only', 'dcn_pd_image_only', 'donut_image_only', 'cmde_image_demo']
    colors = ['#6671FF', '#FF9997', '#C097FF', '#FA97FF', '#97FFAA']
    labels = ['CMDE-\nimg', 'CFRNet-\nimg', 'DCN-PD-\nimg', 'DONUT-\nimg', 'CMDE-\nmul']
    boxplot_data3 = []
    for i in range(len(model_names)):
        boxplot_data3.append(res['eps_ate'][model_names[i]])
    bplot3 = axs[1,0].boxplot(boxplot_data3,
                       sym='', vert=True,
                       patch_artist=True, labels = labels)
    for patch, color in zip(bplot3['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot3['medians']:
        median.set_color('black')
    axs[1,0].set_ylabel('ATE error', fontsize=15)
    axs[1,0].tick_params(axis='both', which='major', labelsize=15)
    axs[1,0].set_facecolor('#F2F2F2')
    axs[1,0].grid(color='white')
    axs[1,0].set_title('ATE error with only image as input for benchmark models', fontsize=15)

    model_names = ['cmde_demo_only', 'cfrnet_wass_demo_only', 'donut_demo_only', 'cmde_image_demo']
    colors = ['#6671FF', '#FF9997', '#FA97FF', '#97FFAA']
    labels = ['CMDE-\ndem', 'CFRNet-\ndem', 'DONUT-\ndem', 'CMDE-\nmul']
    boxplot_data4 = []
    for i in range(len(model_names)):
        boxplot_data4.append(res['eps_ate'][model_names[i]])
    bplot4 = axs[1,1].boxplot(boxplot_data4,
                       sym='', vert=True,
                       patch_artist=True, labels = labels)
    for patch, color in zip(bplot4['boxes'], colors):
        patch.set_facecolor(color)
    for median in bplot4['medians']:
        median.set_color('black')
    axs[1,1].set_ylabel('ATE error', fontsize=15)
    axs[1,1].tick_params(axis='both', which='major', labelsize=15)
    axs[1,1].set_facecolor('#F2F2F2')
    axs[1,1].grid(color='white')
    axs[1,1].set_title('ATE error with only demographic information as input\n for benchmark models', fontsize=15)

    fig.tight_layout(pad=1.0)

    if not os.path.exists('./Figures'):
        os.makedirs('./Figures')
    plt.savefig('./Figures/star_results.pdf', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()