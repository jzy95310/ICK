import sys
sys.path.insert(0, '../../')
import argparse
import numpy as np
import pickle as pkl
from sklearn.ensemble import RandomTreesEmbedding, RandomForestRegressor
from benchmarks.joint_nn import JointDeepViT
from benchmarks.helpers import create_generators_from_data_for_joint_nn
from benchmarks.train_benchmarks import JointNNTrainer
from utils.helpers import calculate_stats, plot_pred_vs_true_vals

import torch
from torch import optim
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
    timestamps_raw, timestamps_raw_train, timestamps_raw_val, timestamps_raw_test = [], [], [], []
    x_train, x_val, x_test, y_train, y_val, y_test = [], [], [], [], [], []
    with open(root_dir, "rb") as fp:
        for data_point in pkl.load(fp):
            sin_t, cos_t = np.sin(2*np.pi*data_point['timestamp']/365), np.cos(2*np.pi*data_point['timestamp']/365)
            timestamps_raw.append([sin_t, cos_t])
            if data_point['timestamp'] < 365:
                x_train.append(data_point['Image'])
                timestamps_raw_train.append([sin_t, cos_t])
                y_train.append(data_point['PM25'])
            elif data_point['timestamp'] < 500:
                x_val.append(data_point['Image'])
                timestamps_raw_val.append([sin_t, cos_t])
                y_val.append(data_point['PM25'])
            else:
                x_test.append(data_point['Image'])
                timestamps_raw_test.append([sin_t, cos_t])
                y_test.append(data_point['PM25'])
    timestamps_raw = np.array(timestamps_raw)
    x_train, timestamps_raw_train, y_train = np.array(x_train), np.array(timestamps_raw_train), np.array(y_train)
    x_val, timestamps_raw_val, y_val = np.array(x_val), np.array(timestamps_raw_val), np.array(y_val)
    x_test, timestamps_raw_test, y_test = np.array(x_test), np.array(timestamps_raw_test), np.array(y_test)

    # Transform timestamps data with Random Trees Embedding Model
    rt_model = RandomTreesEmbedding(n_estimators=300,max_depth=2).fit(timestamps_raw)
    aug_feature_train = rt_model.transform(timestamps_raw_train).toarray()
    aug_feature_val = rt_model.transform(timestamps_raw_val).toarray()
    aug_feature_test = rt_model.transform(timestamps_raw_test).toarray()

    # Train and predict with Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=300,max_depth=2).fit(aug_feature_train, y_train)
    y_pred_train_aug_feature = rf_model.predict(aug_feature_train)
    y_pred_val_aug_feature = rf_model.predict(aug_feature_val)
    y_pred_test_aug_feature = rf_model.predict(aug_feature_test)
    
    # Initialize dataloaders
    img_transform = transforms.ToTensor()
    data_generators = create_generators_from_data_for_joint_nn(
        x_train=x_train, aug_feature_train=aug_feature_train, y_train=y_train, y_pred_train=y_pred_train_aug_feature, 
        x_val=x_val, aug_feature_val=aug_feature_val, y_val=y_val, y_pred_val=y_pred_val_aug_feature, 
        x_test=x_test, aug_feature_test=aug_feature_test, y_test=y_test, y_pred_test=y_pred_test_aug_feature, 
        x_transform=img_transform
    )
    return data_generators, aug_feature_train.shape[1]

def train_joint_model(data_generators, input_width, input_height, patch_size, num_blocks, aug_feature_dim, 
                      lr, weight_decay, epochs, patience, verbose):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = JointDeepViT(
        input_width, 
        input_height,
        patch_size, 
        num_blocks, 
        aug_feature_dim
    )
    optim = 'adam'
    optim_params = {
        'lr': lr,
        # 'momentum': 0.9, 
        'weight_decay': weight_decay
    }
    trainer = JointNNTrainer(
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
    data_generators, aug_feature_dim = preprocess_data()
    y_test_pred, y_test_true = train_joint_model(
        data_generators, 
        args.input_width, 
        args.input_height, 
        args.patch_size, 
        args.num_blocks, 
        aug_feature_dim, 
        args.lr, 
        args.weight_decay, 
        args.epochs, 
        args.patience, 
        args.verbose
    )
    spearmanr, pearsonr, rmse, mae = calculate_stats(
        y_test_pred, 
        y_test_true, 
        data_save_path='./Results/DeepViTRF_seasonal_sorted_by_time.pkl', 
    )
    plot_pred_vs_true_vals(
        y_test_pred, 
        y_test_true, 
        'Predicted PM$_{2.5}$ ($\mu $g m$^{-3}$)', 
        'True PM$_{2.5}$ ($\mu $g m$^{-3}$)',
        fig_save_path='./Figures/DeepViTRF_seasonal_sorted_by_time.pdf', 
        Spearman_R=spearmanr, 
        Pearson_R=pearsonr, 
        RMSE=rmse,
        MAE=mae
    )

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Train a joint DeepViT-RF model on remote sensing data.')
    arg_parser.add_argument('--input_width', type=int, default=224)
    arg_parser.add_argument('--input_height', type=int, default=224)
    arg_parser.add_argument('--patch_size', type=int, default=32)
    arg_parser.add_argument('--num_blocks', type=int, default=2)
    arg_parser.add_argument('--lr', type=float, default=5e-6)
    arg_parser.add_argument('--weight_decay', type=float, default=0.1)
    arg_parser.add_argument('--epochs', type=int, default=250)
    arg_parser.add_argument('--patience', type=int, default=20)
    arg_parser.add_argument('--verbose', type=int, default=1)
    args = arg_parser.parse_known_args()[0]
    main(args)