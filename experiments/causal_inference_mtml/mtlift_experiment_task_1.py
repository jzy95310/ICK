import sys
sys.path.insert(0, '../../')
import argparse
import math
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from causalml.metrics import auuc_score
from model.ick import ICK
from model.cmick import CMICK_MT
from utils.helpers import create_generators_from_data
from utils.losses import FactualMSELoss_MT
from utils.train import CMICKEnsembleTrainer

# Seed for reproducibility
seed = 123
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def main(args):
    train_dir = '../../data/MT-LIFT/train.csv'
    test_dir = '../../data/MT-LIFT/test.csv'
    train_df, test_df = pd.read_csv(train_dir), pd.read_csv(test_dir)
    scaler_train, scaler_test = StandardScaler(), StandardScaler()

    # Training data
    X_train = train_df.to_numpy()[:,3:]
    X_train_scaled = scaler_train.fit_transform(X_train)
    T_train = train_df.to_numpy()[:,2][:,None]
    Y_train_task1 = train_df.to_numpy()[:,0][:,None]

    # Test data
    X_test = test_df.to_numpy()[:,3:]
    X_test_scaled = scaler_test.fit_transform(X_test)
    T_test = test_df.to_numpy()[:,2][:,None]
    Y_test_task1 = test_df.to_numpy()[:,0][:,None]

    # Initialize dataloaders
    data_train = [X_train_scaled, T_train]
    data_test = [X_test_scaled, T_test]
    data_generators_task1 = create_generators_from_data(
        data_train, Y_train_task1, data_test, Y_test_task1, train_batch_size=args.train_batch_size, test_batch_size=1000
    )

    # Model definition of multi-treatment CMDE learner
    n_estimators = args.n_estimators
    n_treatments = len(np.unique(T_train.squeeze()))
    nn_configs = {
        'kernel_assignment': ['ImplicitDenseNetKernel'],
        'kernel_params': {
            'ImplicitDenseNetKernel': {
                'input_dim': X_train_scaled.shape[1],
                'latent_feature_dim': args.nn_width,
                'num_blocks': 1, 
                'num_layers_per_block': args.nn_depth, 
                'num_units': args.nn_width, 
                'activation': args.nn_activation,
            }
        }
    }

    ensemble = []
    for _ in range(n_estimators):
        group_specific_components = [ICK(**nn_configs) for _ in range(n_treatments)]
        shared_components = [ICK(**nn_configs) for _ in range(math.comb(n_treatments, 2))]
        baselearner = CMICK_MT(
            n_treatments=n_treatments,
            group_specific_components=group_specific_components,
            shared_components=shared_components,
            coeff_trainable=True,
        )
        ensemble.append(baselearner)

    # Trainer definition
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = args.optim
    optim_params = {
        'lr': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
    }
    loss_fn = FactualMSELoss_MT()
    trainer = CMICKEnsembleTrainer(
        model=ensemble,
        data_generators=data_generators_task1,
        optim=optim,
        optim_params=optim_params,
        loss_fn=loss_fn,
        model_save_dir=None,
        device=device,
        epochs=args.epochs,
        patience=args.patience, 
        treatment_index=1,
    )
    trainer.train()

    # Compute uplift scores and corresponding indices
    y_test_pred, _, _ = trainer.predict()
    uplift_scores_all = y_test_pred[:, 1:] - y_test_pred[:, [0]]
    max_uplift_indices = np.argmax(uplift_scores_all, axis=1)
    max_uplift_scores = uplift_scores_all[np.arange(uplift_scores_all.shape[0]), max_uplift_indices]
    treatments = np.unique(T_train.squeeze())
    treatments_no_control = treatments[treatments != 0]
    recommended_treatments = treatments_no_control[max_uplift_indices]

    # Prepare dataframe and calculate AUUC
    data = {
        'treatment': T_test.ravel(),
        'outcome': Y_test_task1.ravel(),
        'uplift_score': max_uplift_scores,
        'recommended_treatment': recommended_treatments
    }
    df = pd.DataFrame(data)
    df_filtered = df[(df['treatment'] == 0) | (df['treatment'] == df['recommended_treatment'])].copy()
    df_filtered['binary_treatment'] = (df_filtered['treatment'] == df_filtered['recommended_treatment']).astype(int)
    df_filtered_col = df_filtered[['outcome','binary_treatment','uplift_score']]

    # Calculate AUUC using auuc_score
    auuc = auuc_score(
        df_filtered_col,
        outcome_col='outcome',
        treatment_col='binary_treatment',
        normalize=True
    )
    print("AUUC for Y (click):", auuc)


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Multi-treatment and multi-task causal inference on MT-LIFT')
    arg_parser.add_argument('--train_batch_size', type=int, default=512)
    arg_parser.add_argument('--nn_width', type=int, default=512)
    arg_parser.add_argument('--nn_depth', type=int, default=1)
    arg_parser.add_argument('--nn_activation', type=str, default='relu')
    arg_parser.add_argument('--n_estimators', type=int, default=1)
    arg_parser.add_argument('--optim', type=str, default='sgd')
    arg_parser.add_argument('--lr', type=float, default=1e-5)
    arg_parser.add_argument('--momentum', type=float, default=0.99)
    arg_parser.add_argument('--weight_decay', type=float, default=1e-4)
    arg_parser.add_argument('--epochs', type=int, default=150)
    arg_parser.add_argument('--patience', type=int, default=10)
    args = arg_parser.parse_known_args()[0]
    main(args)