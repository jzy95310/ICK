import sys
sys.path.insert(0, '../../')
import argparse
import numpy as np
import pickle as pkl
from model.ick import ICK, AdditiveICK
from kernels.kernel_fn import sq_exp_kernel_nys, periodic_kernel_nys
from utils.helpers import create_generators_from_data
from utils.train import Trainer
from utils.helpers import calculate_stats, plot_pred_vs_true_vals

import torch
from torchvision import transforms

# To make outputs stable across runs
np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True

def preprocess_data():   
    # Map the timestamps with sine and cosine functions to incorporate seasonality into RF
    root_dir = '../../data/Delhi_labeled_multimodal.pkl'
    X_train, X_val, X_test, y_train, y_val, y_test = [], [], [], [], [], []
    s_train, s_val, s_test = [], [], []
    t_train, t_val, t_test = [], [], []
    with open(root_dir, 'rb') as fp:
        for data_point in pkl.load(fp):
            if data_point['timestamp'] < 365:
                X_train.append(data_point['Image'])
                s_train.append([data_point['lat'], data_point['lon']])
                t_train.append(data_point['timestamp'])
                y_train.append(data_point['PM25'])
            elif data_point['timestamp'] < 500:
                X_val.append(data_point['Image'])
                s_val.append([data_point['lat'], data_point['lon']])
                t_val.append(data_point['timestamp'])
                y_val.append(data_point['PM25'])
            else:
                X_test.append(data_point['Image'])
                s_test.append([data_point['lat'], data_point['lon']])
                t_test.append(data_point['timestamp'])
                y_test.append(data_point['PM25'])
    X_train, s_train, t_train, y_train = np.array(X_train), np.array(s_train), np.array(t_train), np.array(y_train)
    X_val, s_val, t_val, y_val = np.array(X_val), np.array(s_val), np.array(t_val), np.array(y_val)
    X_test, s_test, t_test, y_test = np.array(X_test), np.array(s_test), np.array(t_test), np.array(y_test)
    data_train, data_val, data_test = [X_train, s_train, t_train], [X_val, s_val, t_val], [X_test, s_test, t_test]
    
    # Initialize dataloaders
    img_transform = transforms.ToTensor()
    data_generators = create_generators_from_data(
        data_train, y_train, data_test, y_test, data_val, y_val, x_transform=img_transform
    )
    return data_generators

def train_ick(data_generators, input_width, input_height, patch_size, latent_feature_dim, 
              num_blocks, lr, weight_decay, epochs, patience, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K_x = ICK(
        # Modality 1: image
        kernel_assignment = ['ImplicitDeepViTKernel'], 
        kernel_params = {
            'ImplicitDeepViTKernel': {
                'input_width': input_width,
                'input_height': input_height, 
                'patch_size': patch_size, 
                'latent_feature_dim': latent_feature_dim,
                'num_blocks': num_blocks
            }
        }
    )
    K_s = ICK(
        # Modality 2: spatial information
        kernel_assignment = ['ImplicitNystromKernel'], 
        kernel_params = {
            'ImplicitNystromKernel':{
                'kernel_func': sq_exp_kernel_nys, 
                'params': ['std', 'lengthscale', 'noise'], 
                'vals': [1.,5e-3,0.5],
                'trainable': [True,True,True], 
                'alpha': 1e-5, 
                'num_inducing_points': latent_feature_dim, 
                'nys_space': [[28.40,29.03],[76.92,77.49]]
            }
        }
    )
    K_t = ICK(
        # Modality 3: temporal information
        kernel_assignment = ['ImplicitNystromKernel'], 
        kernel_params = {
            'ImplicitNystromKernel': {
                'kernel_func': periodic_kernel_nys, 
                'params': ['std','period','lengthscale','noise'], 
                'vals': [1.,365.,0.25,0.5], 
                'trainable': [True,True,True,True], 
                'alpha': 1e-5, 
                'num_inducing_points': latent_feature_dim, 
                'nys_space': [[0.,365.]]
            }
        }
    )
    K_xs = ICK(
        kernel_assignment = ['ImplicitDeepViTKernel', 'ImplicitNystromKernel'], 
        kernel_params = {
            'ImplicitDeepViTKernel': {
                'input_width': input_width,
                'input_height': input_height, 
                'patch_size': patch_size, 
                'latent_feature_dim': latent_feature_dim,
                'num_blocks': num_blocks
            }, 
            'ImplicitNystromKernel':{
                'kernel_func': sq_exp_kernel_nys, 
                'params': ['std', 'lengthscale', 'noise'], 
                'vals': [1.,5e-3,0.5],
                'trainable': [True,True,True], 
                'alpha': 1e-5, 
                'num_inducing_points': latent_feature_dim, 
                'nys_space': [[28.40,29.03],[76.92,77.49]]
            }
        }
    )
    K_xt = ICK(
        kernel_assignment = ['ImplicitDeepViTKernel', 'ImplicitNystromKernel'], 
        kernel_params = {
            'ImplicitDeepViTKernel': {
                'input_width': input_width,
                'input_height': input_height, 
                'patch_size': patch_size, 
                'latent_feature_dim': latent_feature_dim,
                'num_blocks': num_blocks
            }, 
            'ImplicitNystromKernel': {
                'kernel_func': periodic_kernel_nys, 
                'params': ['std','period','lengthscale','noise'], 
                'vals': [1.,365.,0.25,0.5], 
                'trainable': [True,True,True,True], 
                'alpha': 1e-5, 
                'num_inducing_points': latent_feature_dim, 
                'nys_space': [[0.,365.]]
            }
        }
    )
    K_st = ICK(
        kernel_assignment = ['ImplicitNystromKernel', 'ImplicitNystromKernel'], 
        kernel_params = {
            'ImplicitNystromKernel':{
                'kernel_func': sq_exp_kernel_nys, 
                'params': ['std', 'lengthscale', 'noise'], 
                'vals': [1.,5e-3,0.5],
                'trainable': [True,True,True], 
                'alpha': 1e-5, 
                'num_inducing_points': latent_feature_dim, 
                'nys_space': [[28.40,29.03],[76.92,77.49]]
            }, 
            'ImplicitNystromKernel': {
                'kernel_func': periodic_kernel_nys, 
                'params': ['std','period','lengthscale','noise'], 
                'vals': [1.,365.,0.25,0.5], 
                'trainable': [True,True,True,True], 
                'alpha': 1e-5, 
                'num_inducing_points': latent_feature_dim, 
                'nys_space': [[0.,365.]]
            }
        }
    )
    K_xst = ICK(
        kernel_assignment = ['ImplicitDeepViTKernel', 'ImplicitNystromKernel', 'ImplicitNystromKernel'],
        kernel_params = {
            'ImplicitDeepViTKernel': {
                'input_width': input_width,
                'input_height': input_height, 
                'patch_size': patch_size, 
                'latent_feature_dim': latent_feature_dim,
                'num_blocks': num_blocks
            },
            'ImplicitNystromKernel':{
                'kernel_func': sq_exp_kernel_nys, 
                'params': ['std', 'lengthscale', 'noise'], 
                'vals': [1.,5e-3,0.5],
                'trainable': [True,True,True], 
                'alpha': 1e-5, 
                'num_inducing_points': latent_feature_dim, 
                'nys_space': [[28.40,29.03],[76.92,77.49]]
            }, 
            'ImplicitNystromKernel': {
                'kernel_func': periodic_kernel_nys, 
                'params': ['std','period','lengthscale','noise'], 
                'vals': [1.,365.,0.25,0.5], 
                'trainable': [True,True,True,True], 
                'alpha': 1e-5, 
                'num_inducing_points': latent_feature_dim, 
                'nys_space': [[0.,365.]]
            }
        }
    )
    model = AdditiveICK(
        components=[K_x,K_s,K_t,K_xs,K_xt,K_st,K_xst], 
        component_assignment=[[0],[1],[2],[0,1],[0,2],[1,2],[0,1,2]], 
        coeffs=[1.,1.,1.,1.,1.,1.,1.], 
        weighted=[False,True,True,False,False,True,False]
    )
    optim = 'sgd'
    optim_params = {
        'lr': lr,
        'momentum': 0.9, 
        'weight_decay': weight_decay
    }
    trainer = Trainer(
        model, 
        data_generators, 
        optim, 
        optim_params, 
        device=device, 
        epochs=epochs, 
        patience=patience, 
        verbose=verbose
    )
    trainer.train()
    return trainer.predict()

def main(args):
    data_generators = preprocess_data()
    y_test_pred, y_test_true = train_ick(
        data_generators, 
        args.input_width, 
        args.input_height, 
        args.patch_size, 
        args.latent_feature_dim, 
        args.num_blocks, 
        args.lr, 
        args.weight_decay, 
        args.epochs, 
        args.patience, 
        args.verbose
    )
    spearmanr, pearsonr, rmse, mae = calculate_stats(
        y_test_pred, 
        y_test_true, 
        data_save_path='./Results/Nonseparable_DeepViT_ICKy_spatiotemporal.pkl', 
    )
    plot_pred_vs_true_vals(
        y_test_pred, 
        y_test_true, 
        'Predicted PM$_{2.5}$ ($\mu $g m$^{-3}$)', 
        'True PM$_{2.5}$ ($\mu $g m$^{-3}$)',
        fig_save_path='./Figures/Nonseparable_DeepViT_ICKy_spatiotemporal.pdf', 
        Spearman_R=spearmanr, 
        Pearson_R=pearsonr, 
        RMSE=rmse,
        MAE=mae
    )

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a nonseparable DeepViT-ICK model on remote sensing data.')
    arg_parser.add_argument('--input_width', type=int, default=224)
    arg_parser.add_argument('--input_height', type=int, default=224)
    arg_parser.add_argument('--patch_size', type=int, default=32)
    arg_parser.add_argument('--num_blocks', type=int, default=2)
    arg_parser.add_argument('--latent_feature_dim', type=int, default=16)
    arg_parser.add_argument('--lr', type=float, default=1e-7)
    arg_parser.add_argument('--weight_decay', type=float, default=0.1)
    arg_parser.add_argument('--epochs', type=int, default=250)
    arg_parser.add_argument('--patience', type=int, default=20)
    arg_parser.add_argument('--verbose', type=int, default=1)
    args = arg_parser.parse_known_args()[0]
    main(args)