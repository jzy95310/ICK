import sys, random
sys.path.insert(0, '../../')
import numpy as np
import pandas as pd
from tqdm.notebook import trange, tqdm
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

import torch
from kernels.nn import ImplicitDenseNetKernel
from model.ick import ICK
from model.cmick import CMICK
from benchmarks.cmgp_modified import CMGP
from benchmarks.cevae_modified import *
from benchmarks.x_learner import X_Learner_RF, X_Learner_BART
from benchmarks.cfrnet import DenseCFRNet
from benchmarks.donut import DenseDONUT
from benchmarks.train_benchmarks import CFRNetTrainer, DONUTTrainer
from utils.train import CMICKEnsembleTrainer
from utils.helpers import *
from utils.losses import *
from utils.metrics import policy_risk
from ganite import Ganite

# To make this notebook's output stable across runs
random.seed(2020)
np.random.seed(2020)
torch.manual_seed(2020)
torch.cuda.manual_seed(2020)
torch.cuda.manual_seed_all(2020)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# 1. Load and preprocess data
def load_and_preprocess_data(train_ratio, test_ratio, random_state=0, in_sample=False):
    randomized_data_dir = '../../data/JOBS/randomized.csv'
    randomized_df = pd.read_csv(randomized_data_dir)
    nonrandomized_data_dir = '../../data/JOBS/nonrandomized.csv'
    nonrandomized_df = pd.read_csv(nonrandomized_data_dir)
    df = pd.concat([randomized_df, nonrandomized_df], ignore_index=True)
    cols_to_normalize = ['Age', 'Education', 'RE75']
    for c in df.columns:
        if c in cols_to_normalize:
            scaler = StandardScaler()
            df[c] = scaler.fit_transform(df[c].to_numpy().reshape(-1,1)).reshape(-1)
    df['RE78'] = df['RE78'].apply(lambda x: 1. if x > 0. else 0.)   # Trasform to binary classification task
    
    N = len(df)
    # Test set only from randomized samples
    test_df = df[:len(randomized_df)].sample(n=int(test_ratio*N), random_state=random_state)  
    df = df.drop(test_df.index)
    train_df = df.sample(n=int(train_ratio*N), random_state=random_state)
    val_df = df.drop(train_df.index)
    X_train, X_val, X_test = train_df.to_numpy()[:,1:-1], val_df.to_numpy()[:,1:-1], test_df.to_numpy()[:,1:-1]
    T_train, T_val, T_test = train_df.to_numpy()[:,:1], val_df.to_numpy()[:,:1], test_df.to_numpy()[:,:1]
    Y_train, Y_val, Y_test = train_df.to_numpy()[:,-1:], val_df.to_numpy()[:,-1:], test_df.to_numpy()[:,-1:]
    
    if in_sample:
        X_train_val = np.concatenate([X_train,X_val], axis=0)
        T_train_val = np.concatenate([T_train,T_val], axis=0)
        Y_train_val = np.concatenate([Y_train,Y_val], axis=0)
        data = {'X_train': X_train, 'T_train': T_train, 'Y_train': Y_train, 'X_val': X_val, 'T_val': T_val,
                'Y_val': Y_val, 'X_test': X_train_val, 'T_test': T_train_val, 'Y_test': Y_train_val}
        data_train, data_val, data_train_val = [X_train, T_train], [X_val, T_val], [X_train_val, T_train_val]
        data_generators = create_generators_from_data(
            x_train=data_train, y_train=Y_train, 
            x_val=data_val, y_val=Y_val,
            x_test=data_train_val, y_test=Y_train_val
        )
    else:
        data = {'X_train': X_train, 'T_train': T_train, 'Y_train': Y_train, 'X_val': X_val, 'T_val': T_val,
                'Y_val': Y_val, 'X_test': X_test, 'T_test': T_test, 'Y_test': Y_test}
        data_train, data_val, data_test = [X_train, T_train], [X_val, T_val], [X_test, T_test]
        data_generators = create_generators_from_data(
            x_train=data_train, y_train=Y_train, 
            x_val=data_val, y_val=Y_val,
            x_test=data_test, y_test=Y_test
        )
    return data_generators, data


# 2. Define CMDE model
def build_cmnn_ensemble(input_dim, load_weights=False):
    alpha11, alpha12, alpha13 = 1.0, 1.0, 0.1
    alpha21, alpha22, alpha23 = 1.0, 1.0, 0.1
    num_estimators = 10

    ensemble, ensemble_weights = [], {}
    for i in range(num_estimators):
        f11 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim,
                    'latent_feature_dim': 512,
                    'num_blocks': 1, 
                    'num_layers_per_block': 1, 
                    'num_units': 512, 
                    'activation': 'tanh'
                }
            }
        )
        f12 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim,
                    'latent_feature_dim': 512,
                    'num_blocks': 1, 
                    'num_layers_per_block': 1, 
                    'num_units': 512, 
                    'activation': 'tanh'
                }
            }
        )
        f13 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim,
                    'latent_feature_dim': 512,
                    'num_blocks': 1, 
                    'num_layers_per_block': 1, 
                    'num_units': 512, 
                    'activation': 'tanh'
                }
            }
        )
        f21 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim,
                    'latent_feature_dim': 512,
                    'num_blocks': 1, 
                    'num_layers_per_block': 1, 
                    'num_units': 512, 
                    'activation': 'tanh'
                }
            }
        )
        f22 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim,
                    'latent_feature_dim': 512,
                    'num_blocks': 1, 
                    'num_layers_per_block': 1, 
                    'num_units': 512, 
                    'activation': 'tanh'
                }
            }
        )
        f23 = ICK(
            kernel_assignment=['ImplicitDenseNetKernel'],
            kernel_params={
                'ImplicitDenseNetKernel':{
                    'input_dim': input_dim,
                    'latent_feature_dim': 512,
                    'num_blocks': 1, 
                    'num_layers_per_block': 1, 
                    'num_units': 512, 
                    'activation': 'tanh'
                }
            }
        )
        if load_weights:
            for f in ['f11', 'f12', 'f13', 'f21', 'f22', 'f23']:
                eval(f).kernels[0].load_state_dict(torch.load('./checkpoints/cmick_jobs.pt')['model_'+str(i+1)][f])
        else:
            model_weights = {
                'f11': f11.kernels[0].state_dict(), 'f12': f12.kernels[0].state_dict(), 'f13': f13.kernels[0].state_dict(), 
                'f21': f21.kernels[0].state_dict(), 'f22': f22.kernels[0].state_dict(), 'f23': f23.kernels[0].state_dict()
            }
            ensemble_weights['model_'+str(i+1)] = model_weights
        # Set output_binary=True for binary Y0 and Y1
        baselearner = CMICK(
            control_components=[f11,f21], treatment_components=[f12,f22], shared_components=[f13,f23],
            control_coeffs=[alpha11,alpha21], treatment_coeffs=[alpha12,alpha22], shared_coeffs=[alpha13,alpha23], 
            coeff_trainable=True, output_binary=True
        )
        ensemble.append(baselearner)
    if not load_weights:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(ensemble_weights, './checkpoints/cmick_jobs.pt')

    return ensemble


# 3. Training and evaluation of CMNN model
def fit_and_evaluate_cmnn(ensemble, data_generators_in_sample, data_generators_out_sample, data_in_sample, 
                          data_out_sample, lr, treatment_index=1): 
    # The index of "T_train" in "data_train" is 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 20
    trainer = CMICKEnsembleTrainer(
        model=ensemble,
        data_generators=data_generators_in_sample,
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
    y_test_in_sample, t_test_in_sample = data_in_sample['Y_test'], data_in_sample['T_test']
    r_pol_in_sample = policy_risk(mean_test_pred, y_test_in_sample, t_test_in_sample)
    
    trainer.data_generators = data_generators_out_sample
    mean_test_pred, std_test_pred, y_test_true = trainer.predict()
    y_test_out_sample, t_test_out_sample = data_out_sample['Y_test'], data_out_sample['T_test']
    r_pol_out_sample = policy_risk(mean_test_pred, y_test_out_sample, t_test_out_sample)
    
    print('Policy risk (CMNN, in-sample):             %.4f' % (r_pol_in_sample))
    print('Policy risk (CMNN, out-of-sample):             %.4f' % (r_pol_out_sample))
    return r_pol_in_sample, r_pol_out_sample


# 4. Benchmark 1: original CMGP
def fit_and_evaluate_original_cmgp(data_in_sample, data_out_sample):
    X_train, T_train, Y_train = data_in_sample['X_train'], data_in_sample['T_train'], data_in_sample['Y_train']
    X_test, T_test, Y_test = data_in_sample['X_test'], data_in_sample['T_test'], data_in_sample['Y_test']
    cmgp_model = CMGP(X_train, T_train, Y_train)
    mu0_test_pred, mu1_test_pred = cmgp_model.predict(X_test, return_var=False)
    mu_test_pred_in_sample = np.concatenate([mu0_test_pred, mu1_test_pred], axis=1)
    r_pol_in_sample = policy_risk(mu_test_pred_in_sample, Y_test, T_test)
    
    X_test, T_test, Y_test = data_out_sample['X_test'], data_out_sample['T_test'], data_out_sample['Y_test']
    mu0_test_pred, mu1_test_pred = cmgp_model.predict(X_test, return_var=False)
    mu_test_pred_out_sample = np.concatenate([mu0_test_pred, mu1_test_pred], axis=1)
    r_pol_out_sample = policy_risk(mu_test_pred_out_sample, Y_test, T_test)
    
    print('Policy risk (CMGP, in-sample):             %.4f' % (r_pol_in_sample))
    print('Policy risk (CMGP, out-of-sample):             %.4f' % (r_pol_out_sample))
    return r_pol_in_sample, r_pol_out_sample


# 5. Benchmark 2: CEVAE
def fit_and_evaluate_cevae(data_in_sample, data_out_sample):
    lr = 1e-4
    weight_decay = 1e-4
    batch_size = int(data_in_sample['X_train'].shape[0]/8)
    train_iters = 20
    eval_iters = 200
    latent_dim = 20
    n_h = 64
    X_train, T_train, Y_train, X_test = torch.tensor(data_in_sample['X_train']).float(), torch.tensor(data_in_sample['T_train']).float(), \
                                        torch.tensor(data_in_sample['Y_train']).float(), torch.tensor(data_in_sample['X_test']).float()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    X_train = X_train.to(device)
    T_train = T_train.to(device)
    Y_train = Y_train.to(device)
    X_test = X_test.to(device)
    
    # init networks (overwritten per replication)
    p_x_z_dist = p_x_z(dim_in=latent_dim, nh=3, dim_h=n_h).to(device)
    p_t_z_dist = p_t_z(dim_in=latent_dim, nh=1, dim_h=n_h, dim_out=1).to(device)
    p_y_zt_dist = p_y_zt(dim_in=latent_dim, nh=3, dim_h=n_h, dim_out=1, output_binary=True).to(device)
    q_t_x_dist = q_t_x(dim_in=X_train.shape[1], nh=1, dim_h=n_h, dim_out=1).to(device)

    # t is not feed into network, therefore not increasing input size (y is fed).
    q_y_xt_dist = q_y_xt(dim_in=X_train.shape[1], nh=3, dim_h=n_h, dim_out=1, output_binary=True).to(device)
    q_z_tyx_dist = q_z_tyx(dim_in=X_train.shape[1]+1, nh=3, dim_h=n_h, dim_out=latent_dim).to(device)
    p_z_dist = normal.Normal(torch.zeros(latent_dim).to(device), torch.ones(latent_dim).to(device))

    # Create optimizer
    params = list(p_x_z_dist.parameters()) + \
             list(p_t_z_dist.parameters()) + \
             list(p_y_zt_dist.parameters()) + \
             list(q_t_x_dist.parameters()) + \
             list(q_y_xt_dist.parameters()) + \
             list(q_z_tyx_dist.parameters())

    # Adam is used, like original implementation, in paper Adamax is suggested
    optimizer = optim.Adamax(params, lr=lr, weight_decay=weight_decay)

    # init q_z inference
    q_z_tyx_dist = init_qz(q_z_tyx_dist, Y_train, T_train, X_train).to(device)

    # Training
    loss = []
    for _ in tqdm(range(train_iters), position=0, leave=True):
        i = np.random.choice(X_train.shape[0],size=batch_size,replace=False)
        Y_train_shuffled = Y_train[i,:].to(device)
        X_train_shuffled = X_train[i,:].to(device)
        T_train_shuffled = T_train[i,:].to(device)

        # inferred distribution over z
        xy = torch.cat((X_train_shuffled, Y_train_shuffled), 1).to(device)
        z_infer = q_z_tyx_dist(xy=xy, t=T_train_shuffled)
        # use a single sample to approximate expectation in lowerbound
        z_infer_sample = z_infer.sample()        

        # RECONSTRUCTION LOSS
        # p(x|z)
        x_con = p_x_z_dist(z_infer_sample)
        # l1 = x_bin.log_prob(x_train).sum(1)

        l2 = x_con.log_prob(X_train_shuffled).sum(1)

        # p(t|z)
        t = p_t_z_dist(z_infer_sample)
        l3 = t.log_prob(T_train_shuffled).squeeze()

        # p(y|t,z)
        # for training use trt_train, in out-of-sample prediction this becomes t_infer
        y = p_y_zt_dist(z_infer_sample, T_train_shuffled)
        l4 = y.log_prob(Y_train_shuffled).squeeze()

        # REGULARIZATION LOSS
        # p(z) - q(z|x,t,y)
        # approximate KL
        l5 = (p_z_dist.log_prob(z_infer_sample) - z_infer.log_prob(z_infer_sample)).sum(1)

        # AUXILIARY LOSS
        # q(t|x)
        t_infer = q_t_x_dist(X_train_shuffled)
        l6 = t_infer.log_prob(T_train_shuffled).squeeze()

        # q(y|x,t)
        y_infer = q_y_xt_dist(X_train_shuffled, T_train_shuffled)
        l7 = y_infer.log_prob(Y_train_shuffled).squeeze()

        # Total objective
        # inner sum to calculate loss per item, torch.mean over batch
        loss_mean = torch.mean(l2 + l3 + l4 + l5 + l6 + l7)
        loss.append(loss_mean.cpu().detach().numpy())
        objective = -loss_mean

        optimizer.zero_grad()
        # Calculate gradients
        objective.backward()
        # Update step
        optimizer.step()
        
    # Evaluation for in-sample
    Y0_pred, Y1_pred = [], []
    t_infer = q_t_x_dist(X_test)

    eval_iters = 1000
    for _ in tqdm(range(eval_iters), position=0, leave=True):
        ttmp = t_infer.sample()
        y_infer = q_y_xt_dist(X_test, ttmp)

        xy = torch.cat((X_test, y_infer.sample()), 1)
        z_infer = q_z_tyx_dist(xy=xy, t=ttmp).sample()
        y0 = p_y_zt_dist(z_infer, torch.zeros(z_infer.shape[0],1).to(device)).sample()
        y1 = p_y_zt_dist(z_infer, torch.ones(z_infer.shape[0],1).to(device)).sample()
        Y0_pred.append(y0.detach().cpu().numpy().ravel())
        Y1_pred.append(y1.detach().cpu().numpy().ravel())

    # sameple from the treated and control    
    mu0_test_pred = np.mean(np.array(Y0_pred), axis=0)
    mu1_test_pred = np.mean(np.array(Y1_pred), axis=0)
    y_test_in_sample, t_test_in_sample = data_in_sample['Y_test'], data_in_sample['T_test']
    mu_test_pred_in_sample = np.stack([mu0_test_pred, mu1_test_pred], axis=1)
    r_pol_in_sample = policy_risk(mu_test_pred_in_sample, y_test_in_sample, t_test_in_sample)
    
    # Evaluation for out-of-sample
    X_test = torch.tensor(data_out_sample['X_test']).float().to(device)
    Y0_pred, Y1_pred = [], []
    t_infer = q_t_x_dist(X_test)

    eval_iters = 1000
    for _ in tqdm(range(eval_iters), position=0, leave=True):
        ttmp = t_infer.sample()
        y_infer = q_y_xt_dist(X_test, ttmp)

        xy = torch.cat((X_test, y_infer.sample()), 1)
        z_infer = q_z_tyx_dist(xy=xy, t=ttmp).sample()
        y0 = p_y_zt_dist(z_infer, torch.zeros(z_infer.shape[0],1).to(device)).sample()
        y1 = p_y_zt_dist(z_infer, torch.ones(z_infer.shape[0],1).to(device)).sample()
        Y0_pred.append(y0.detach().cpu().numpy().ravel())
        Y1_pred.append(y1.detach().cpu().numpy().ravel())

    # sameple from the treated and control    
    mu0_test_pred = np.mean(np.array(Y0_pred), axis=0)
    mu1_test_pred = np.mean(np.array(Y1_pred), axis=0)
    y_test_out_sample, t_test_out_sample = data_out_sample['Y_test'], data_out_sample['T_test']
    mu_test_pred_out_sample = np.stack([mu0_test_pred, mu1_test_pred], axis=1)
    r_pol_out_sample = policy_risk(mu_test_pred_out_sample, y_test_out_sample, t_test_out_sample)
    
    print('Policy risk (CEVAE, in-sample):             %.4f' % (r_pol_in_sample))
    print('Policy risk (CEVAE, out-of-sample):             %.4f' % (r_pol_out_sample))
    return r_pol_in_sample, r_pol_out_sample


# 6. Benchmark 3: GANITE
def fit_and_evaluate_ganite(data_in_sample, data_out_sample):
    X_train, T_train, Y_train, X_test = data_in_sample['X_train'], data_in_sample['T_train'], \
                                        data_in_sample['Y_train'], data_in_sample['X_test']
    Y_test_in_sample, T_test_in_sample = data_in_sample['Y_test'], data_in_sample['T_test']
    model = Ganite(X_train, T_train, Y_train, dim_hidden=4, alpha=1, beta=5, minibatch_size=128, depth=3)
    with torch.no_grad():
        X_test = model._check_tensor(X_test).float()
        y_test_pred = model.inference_nets(X_test).detach().cpu().numpy()
    r_pol_in_sample = policy_risk(y_test_pred, Y_test_in_sample, T_test_in_sample)
    
    X_test = data_out_sample['X_test']
    Y_test_out_sample, T_test_out_sample = data_out_sample['Y_test'], data_out_sample['T_test']
    with torch.no_grad():
        X_test = model._check_tensor(X_test).float()
        y_test_pred = model.inference_nets(X_test).detach().cpu().numpy()
    r_pol_out_sample = policy_risk(y_test_pred, Y_test_out_sample, T_test_out_sample)
    
    print('Policy risk (GANITE, in-sample):             %.4f' % (r_pol_in_sample))
    print('Policy risk (GANITE, out-of-sample):             %.4f' % (r_pol_out_sample))
    return r_pol_in_sample, r_pol_out_sample


# 7. Benchmark 4: X-learner-RF
def fit_and_evaluate_x_learner_rf(data_in_sample, data_out_sample):
    X_train, T_train, Y_train, X_test = data_in_sample['X_train'], data_in_sample['T_train'], \
                                        data_in_sample['Y_train'], data_in_sample['X_test']
    x_learner_rf = X_Learner_RF()
    x_learner_rf.fit(X_train, T_train, Y_train)
    mu_test_pred_in_sample = x_learner_rf.predict(X_test)
    y_test_in_sample, t_test_in_sample = data_in_sample['Y_test'], data_in_sample['T_test']
    r_pol_in_sample = policy_risk(mu_test_pred_in_sample, y_test_in_sample, t_test_in_sample)
    
    X_test = data_out_sample['X_test']
    mu_test_pred_out_sample = x_learner_rf.predict(X_test)
    y_test_out_sample, t_test_out_sample = data_out_sample['Y_test'], data_out_sample['T_test']
    r_pol_out_sample = policy_risk(mu_test_pred_out_sample, y_test_out_sample, t_test_out_sample)
    
    print('Policy risk (X-learner-RF, in-sample):             %.4f' % (r_pol_in_sample))
    print('Policy risk (X-learner-RF, out-of-sample):             %.4f' % (r_pol_out_sample))
    return r_pol_in_sample, r_pol_out_sample


# 8. Benchmark 5: X-learner-BART
def fit_and_evaluate_x_learner_bart(data_in_sample, data_out_sample):
    X_train, T_train, Y_train, X_test = data_in_sample['X_train'], data_in_sample['T_train'], \
                                        data_in_sample['Y_train'], data_in_sample['X_test']
    x_learner_bart = X_Learner_BART(n_trees=20)
    x_learner_bart.fit(X_train, T_train, Y_train)
    mu_test_pred_in_sample = x_learner_bart.predict(X_test)
    y_test_in_sample, t_test_in_sample = data_in_sample['Y_test'], data_in_sample['T_test']
    r_pol_in_sample = policy_risk(mu_test_pred_in_sample, y_test_in_sample, t_test_in_sample)
    
    X_test = data_out_sample['X_test']
    mu_test_pred_out_sample = x_learner_bart.predict(X_test)
    y_test_out_sample, t_test_out_sample = data_out_sample['Y_test'], data_out_sample['T_test']
    r_pol_out_sample = policy_risk(mu_test_pred_out_sample, y_test_out_sample, t_test_out_sample)
    
    print('Policy risk (X-learner-BART, in-sample):             %.4f' % (r_pol_in_sample))
    print('Policy risk (X-learner-BART, out-of-sample):             %.4f' % (r_pol_out_sample))
    return r_pol_in_sample, r_pol_out_sample


# 9. Benchmark 6: CFRNet
def fit_and_evaluate_cfrnet(input_dim, phi_depth, phi_width, h_depth, h_width, data_generators_in_sample, 
                            data_generators_out_sample, data_in_sample, data_out_sample, lr, alpha, 
                            metric='W2', treatment_index=1, load_weights=False):
    cfrnet = DenseCFRNet(input_dim, phi_depth, phi_width, h_depth, h_width, activation='tanh')
    if load_weights:
        cfrnet.load_state_dict(torch.load('./checkpoints/cfrnet_jobs.pt'))
    else:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(cfrnet.state_dict(), './checkpoints/cfrnet_jobs.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 20
    trainer = CFRNetTrainer(
        model=cfrnet,
        data_generators=data_generators_in_sample,
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
    y_test_in_sample, t_test_in_sample = data_in_sample['Y_test'], data_in_sample['T_test']
    r_pol_in_sample = policy_risk(y_test_pred, y_test_in_sample, t_test_in_sample)
    
    trainer.data_generators = data_generators_out_sample
    y_test_pred, y_test_true = trainer.predict()
    y_test_out_sample, t_test_out_sample = data_out_sample['Y_test'], data_out_sample['T_test']
    r_pol_out_sample = policy_risk(y_test_pred, y_test_out_sample, t_test_out_sample)
    
    print('Policy risk (CFRNet, in-sample):             %.4f' % (r_pol_in_sample))
    print('Policy risk (CFRNet, out-of-sample):             %.4f' % (r_pol_out_sample))
    return r_pol_in_sample, r_pol_out_sample


# 10. Benchmark 7: DONUT
def fit_and_evaluate_donut(input_dim, phi_depth, phi_width, h_depth, h_width, data_generators_in_sample, 
                            data_generators_out_sample, data_in_sample, data_out_sample, lr, treatment_index=1, 
                           load_weights=False):
    donut = DenseDONUT(input_dim, phi_depth, phi_width, h_depth, h_width, activation='tanh')
    if load_weights:
        donut.load_state_dict(torch.load('./checkpoints/donut_jobs.pt'))
    else:
        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        torch.save(donut.state_dict(), './checkpoints/donut_jobs.pt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optim = 'sgd'
    optim_params = {
        'lr': lr, 
        'momentum': 0.99,
        'weight_decay': 1e-4
    }
    epochs, patience = 1000, 20
    trainer = DONUTTrainer(
        model=donut,
        data_generators=data_generators_in_sample,
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
    y_test_in_sample, t_test_in_sample = data_in_sample['Y_test'], data_in_sample['T_test']
    r_pol_in_sample = policy_risk(y_test_pred, y_test_in_sample, t_test_in_sample)
    
    trainer.data_generators = data_generators_out_sample
    y_test_pred, y_test_true = trainer.predict()
    y_test_out_sample, t_test_out_sample = data_out_sample['Y_test'], data_out_sample['T_test']
    r_pol_out_sample = policy_risk(y_test_pred, y_test_out_sample, t_test_out_sample)
    
    print('Policy risk (DONUT, in-sample):             %.4f' % (r_pol_in_sample))
    print('Policy risk (DONUT, out-of-sample):             %.4f' % (r_pol_out_sample))
    return r_pol_in_sample, r_pol_out_sample


# Main function
def main():
    train_ratio, test_ratio, n_iters = 0.56, 0.20, 10
    res = {'in-sample': defaultdict(list), 'out-sample': defaultdict(list)}
    for i in trange(n_iters):
        print("Iteration {}".format(i+1))
        data_generators_in_sample, data_in_sample = load_and_preprocess_data(
            train_ratio, test_ratio, random_state=i, in_sample=True)
        data_generators_out_sample, data_out_sample = load_and_preprocess_data(
            train_ratio, test_ratio, random_state=i, in_sample=False)
        input_dim = data_in_sample['X_train'].shape[1]
        # Make sure CMDE has the same starting point for each experimental run
        ensemble = build_cmnn_ensemble(input_dim, load_weights=(i!=0))
        r_pol_cmnn_in_sample, r_pol_cmnn_out_sample = fit_and_evaluate_cmnn(
            ensemble, data_generators_in_sample, data_generators_out_sample, data_in_sample, 
            data_out_sample, lr=2e-3)
        res['in-sample']['r_pol_cmnn'].append(r_pol_cmnn_in_sample)
        res['out-sample']['r_pol_cmnn'].append(r_pol_cmnn_out_sample)
        r_pol_cmgp_in_sample, r_pol_cmgp_out_sample = fit_and_evaluate_original_cmgp(
            data_in_sample, data_out_sample)
        res['in-sample']['r_pol_cmgp'].append(r_pol_cmgp_in_sample)
        res['out-sample']['r_pol_cmgp'].append(r_pol_cmgp_out_sample)
        r_pol_cevae_in_sample, r_pol_cevae_out_sample = fit_and_evaluate_cevae(
            data_in_sample, data_out_sample)
        res['in-sample']['r_pol_cevae'].append(r_pol_cevae_in_sample)
        res['out-sample']['r_pol_cevae'].append(r_pol_cevae_out_sample)
        r_pol_ganite_in_sample, r_pol_ganite_out_sample  = fit_and_evaluate_ganite(
            data_in_sample, data_out_sample)
        res['in-sample']['r_pol_ganite'].append(r_pol_ganite_in_sample)
        res['out-sample']['r_pol_ganite'].append(r_pol_ganite_out_sample)
#         sqrt_pehe_x_learner_rf_in_sample, sqrt_pehe_x_learner_rf_out_sample = fit_and_evaluate_x_learner_rf(
#             data_in_sample, data_out_sample)
#         res['in-sample']['sqrt_pehe_x_learner_rf'].append(sqrt_pehe_x_learner_rf_in_sample)
#         res['out-sample']['sqrt_pehe_x_learner_rf'].append(sqrt_pehe_x_learner_rf_out_sample)
#         sqrt_pehe_x_learner_bart_in_sample, sqrt_pehe_x_learner_bart_out_sample = fit_and_evaluate_x_learner_bart(
#             data_in_sample, data_out_sample)
#         res['in-sample']['sqrt_pehe_x_learner_bart'].append(sqrt_pehe_x_learner_bart_in_sample)
#         res['out-sample']['sqrt_pehe_x_learner_bart'].append(sqrt_pehe_x_learner_bart_out_sample)
        r_pol_cfrnet_wass_in_sample, r_pol_cfrnet_wass_out_sample = fit_and_evaluate_cfrnet(
            input_dim, 2, 512, 2, 512, data_generators_in_sample, data_generators_out_sample, 
            data_in_sample, data_out_sample, lr=1e-4, alpha=1, metric='W2', 
            treatment_index=1, load_weights=(i!=0))
        res['in-sample']['r_pol_cfrnet_wass'].append(r_pol_cfrnet_wass_in_sample)
        res['out-sample']['r_pol_cfrnet_wass'].append(r_pol_cfrnet_wass_out_sample)
        r_pol_cfrnet_mmd_in_sample, r_pol_cfrnet_mmd_out_sample = fit_and_evaluate_cfrnet(
            input_dim, 2, 512, 2, 512, data_generators_in_sample, data_generators_out_sample, 
            data_in_sample, data_out_sample, lr=1e-4, alpha=1, metric='MMD', 
            treatment_index=1, load_weights=(i!=0))
        res['in-sample']['r_pol_cfrnet_mmd'].append(r_pol_cfrnet_mmd_in_sample)
        res['out-sample']['r_pol_cfrnet_mmd'].append(r_pol_cfrnet_mmd_out_sample)
        r_pol_donut_in_sample, r_pol_donut_out_sample = fit_and_evaluate_donut(
            input_dim, 2, 512, 2, 512, data_generators_in_sample, data_generators_out_sample, 
            data_in_sample, data_out_sample, lr=1e-4, treatment_index=1, load_weights=(i!=0))
        res['in-sample']['r_pol_donut'].append(r_pol_donut_in_sample)
        res['out-sample']['r_pol_donut'].append(r_pol_donut_out_sample)
    try:
        os.makedirs('./results')
    except FileExistsError:
        print('Directory already exists.')
    with open('./results/jobs_results.pkl', 'wb') as fp:
        pkl.dump(res, fp)
    
    for s in [True, False]:
        print("Setting: {}".format("In-sample" if s else "Out-of-sample"))
        s_str = 'in-sample' if s else 'out-sample'
        for k in res[s_str].keys():
            method = k.split('_pol_')[-1]
            pehe_mean, pehe_std = np.mean(res[s_str][k]), np.std(res[s_str][k])
            print('Policy risk ({}):             {:.4f} +/- {:4f}'.format(method, pehe_mean, pehe_std))

if __name__ == "__main__":
    main()