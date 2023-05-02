import sys
sys.path.insert(0, '../../')
import argparse
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from model.ick import ICK
from utils.helpers import create_generators_from_data
from kernels.kernel_fn import matern_type1_kernel_nys, periodic_kernel_nys
from utils.train import EnsembleTrainer

# To make outputs stable across runs
np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True

def preprocess_data(batch_size, random_state):
    df = pd.read_csv('../../data/UCI_ml_repo/HouseElectric.txt', sep=';', dtype=str)
    df = df[df['Global_active_power'] != '?']
    N = len(df)
    df['Time'] = df['Time'].apply(lambda x: int((pd.Timestamp(x) - pd.Timestamp('00:00:00')).total_seconds()/60))
    df['Date'] = df['Date'].apply(lambda x: '{}-{}-{}'.format(x.split('/')[2], x.split('/')[1], x.split('/')[0]))
    df['Date'] = df['Date'].apply(lambda x: (pd.Timestamp(x) - pd.Timestamp('2006-12-16')).days)
    df['T'] = df.apply(lambda row: row['Date']*1440 + row['Time'], axis=1)
    df = df.drop(['Date','Time'], axis=1)
    for col in df.columns[:-1]:
        df[col] = df[col].apply(lambda x: float(x))
#     df = df[list(df.columns[:2]) + list(df.columns[3:]) + [df.columns[2]]]
    df = df[list(df.columns[1:-1]) + [df.columns[-1]] + [df.columns[0]]]
    df_train = df.sample(n=int(N*4/9), replace=False, random_state=random_state)
    df = df.drop(df_train.index, axis=0)
    df_val = df.sample(n=int(N*2/9), replace=False, random_state=random_state)
    df_test = df.drop(df_val.index, axis=0)
    X_train, t_train, y_train = np.array(df_train[df_train.columns[:-2]]), np.array(df_train[df_train.columns[-2]]), np.array(df_train[df_train.columns[-1]])
    X_val, t_val, y_val = np.array(df_val[df_val.columns[:-2]]), np.array(df_val[df_val.columns[-2]]), np.array(df_val[df_val.columns[-1]])
    X_test, t_test, y_test = np.array(df_test[df_test.columns[:-2]]), np.array(df_test[df_test.columns[-2]]), np.array(df_test[df_test.columns[-1]])

    for j in range(X_train.shape[1]):
        scaler = StandardScaler()
        X_train[:,j] = scaler.fit_transform(X_train[:,j].reshape(-1,1)).squeeze()
        X_val[:,j] = scaler.transform(X_val[:,j].reshape(-1,1)).squeeze()
        X_test[:,j] = scaler.transform(X_test[:,j].reshape(-1,1)).squeeze()
    
    y_train = scaler.fit_transform(y_train.reshape(-1,1)).squeeze()
    y_val = scaler.transform(y_val.reshape(-1,1)).squeeze()
    y_test = scaler.transform(y_test.reshape(-1,1)).squeeze()

    data_generators = create_generators_from_data(
        [X_train,t_train], y_train, [X_test,t_test], y_test, [X_val,t_val], y_val, 
        train_batch_size=batch_size, val_batch_size=batch_size, test_batch_size=batch_size
    )
    return data_generators

def train_ick_ensemble(data_generators, input_dim, depth, width, latent_feature_dim, lr, weight_decay, 
                       epochs, patience, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    kernel_assignment = ['ImplicitDenseNetKernel', 'ImplicitNystromKernel']
    kernel_params = { 
        'ImplicitDenseNetKernel': {
            'input_dim': input_dim, 
            'latent_feature_dim': latent_feature_dim, 
            'num_blocks': depth,
            'num_layers_per_block': 1, 
            'num_units': width,
        }, 
        'ImplicitNystromKernel': {
            'kernel_func': periodic_kernel_nys, 
            'params': ['std','period','lengthscale','noise'], 
            'vals': [1., 1440., 1., 0.5], 
            'trainable': [True,True,True,True], 
            'alpha': 1e-5, 
            'num_inducing_points': latent_feature_dim, 
            'nys_space': [[0.,1440.]]
        }
    }
    ensemble = [ICK(kernel_assignment, kernel_params) for _ in range(10)]
    optim = 'sgd'
    optim_params = {
        'lr': lr,
        'momentum': 0.9, 
        'weight_decay': weight_decay
    }
    trainer = EnsembleTrainer(
        ensemble, 
        data_generators, 
        optim, 
        optim_params, 
        num_jobs=1, 
        device=device, 
        epochs=epochs, 
        patience=patience, 
        verbose=verbose
    )
    trainer.train()
    return trainer.predict()

def main(args):
    random_states, eps = [42, 43, 44], 1e-6
    rmse_arr, nll_arr = [], []
    for random_state in random_states:
        print("random state = {}".format(random_state))
        data_generators = preprocess_data(args.batch_size, random_state=random_state)
        y_test_pred_mean, y_test_pred_std, y_test_true = train_ick_ensemble(
            data_generators, 
            args.input_dim, 
            args.depth, 
            args.width, 
            args.latent_feature_dim, 
            args.lr, 
            args.weight_decay, 
            args.epochs, 
            args.patience, 
            args.verbose
        )
        rmse = np.sqrt(np.mean((y_test_pred_mean - y_test_true)**2))
        nll = np.mean(0.5 * (np.log(np.maximum(y_test_pred_std**2, eps)) + \
                     ((y_test_pred_mean - y_test_true)**2)/np.maximum(y_test_pred_std**2, eps)))
        print("RMSE (ICKy) = {:.4f}".format(rmse))
        print("NLL (ICKy) = {:.4f}".format(nll))
        rmse_arr.append(rmse)
        nll_arr.append(nll)
    print("Final results:")
    print("Final RMSE (ICKy) = {:.4f} +/- {:.4f}".format(np.mean(rmse), np.std(rmse)))
    print("Final NLL (ICKy) = {:.4f} +/- {:.4f}".format(np.mean(nll), np.std(nll)))

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train an ICK ensemble model on HouseElectric data.')
    arg_parser.add_argument('--input_dim', type=int, default=6)
    arg_parser.add_argument('--depth', type=int, default=2)
    arg_parser.add_argument('--width', type=int, default=50)
    arg_parser.add_argument('--latent_feature_dim', type=int, default=16)
    arg_parser.add_argument('--lr', type=float, default=1e-4)
    arg_parser.add_argument('--batch_size', type=int, default=512)
    arg_parser.add_argument('--weight_decay', type=float, default=0)
    arg_parser.add_argument('--epochs', type=int, default=100)
    arg_parser.add_argument('--patience', type=int, default=15)
    arg_parser.add_argument('--verbose', type=int, default=0)
    args = arg_parser.parse_known_args()[0]
    main(args)
