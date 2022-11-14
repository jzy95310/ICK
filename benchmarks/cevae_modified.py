# Original post: https://github.com/rik-helwegen/CEVAE_pytorch/
# Author: Rik Helwegen
# ArXiv: https://arxiv.org/abs/1705.08821
# Reference: Louizos, Christos, et al. "Causal effect inference with deep latent-variable models." 
# Advances in neural information processing systems 30 (2017).

import numpy as np

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.distributions import bernoulli, normal
import torch.distributions

# class naming convention: p_A_BC -> p(A|B,C)

def init_qz(qz, y,t,x):
    """
    Initialize qz towards outputting standard normal distributions
    - with standard torch init of weights the gradients tend to explode after first update step
    """
    idx = list(range(x.shape[0]))
    np.random.shuffle(idx)

    optimizer = optim.Adam(qz.parameters(), lr=0.001)
    
    for i in range(50):
        batch = np.random.choice(idx, 1)
        x_train, y_train, t_train = (x[batch]), (y[batch]), (t[batch])
        xy = torch.cat((x_train, y_train), 1)

        z_infer = qz(xy=xy, t=t_train)

        # no binary covariates here
        # KL(q_z|p_z) mean approx, to be minimized
        # KLqp = (z_infer.log_prob(z_infer.mean) - pz.log_prob(z_infer.mean)).sum(1)
        # Analytic KL
        KLqp = (-torch.log(z_infer.stddev) + 1/2*(z_infer.variance + z_infer.mean**2 - 1)).sum(1)

        objective = KLqp
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()

        if KLqp != KLqp:
            raise ValueError('KL(pz,qz) contains NaN during init')

    return qz

class p_x_z(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out_bin=0, dim_out_con=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out_bin = dim_out_bin
        self.dim_out_con = dim_out_con

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh-1)])
        # output layer defined separate for continuous and binary outputs
        self.output_bin = nn.Linear(dim_h, dim_out_bin)
        # for each output an mu and sigma are estimated
        self.output_con_mu = nn.Linear(dim_h, dim_out_con)
        self.output_con_sigma = nn.Linear(dim_h, dim_out_con)
        self.softplus = nn.Softplus()

    def forward(self, z_input):
        z = F.elu(self.input(z_input))
        for i in range(self.nh-1):
            z = F.elu(self.hidden[i](z))
        # for binary outputs:
#         x_bin_p = torch.sigmoid(self.output_bin(z))
#         x_bin = bernoulli.Bernoulli(x_bin_p)
        # for continuous outputs
        mu, sigma = self.output_con_mu(z), self.softplus(self.output_con_sigma(z))
        x_con = normal.Normal(mu, sigma)

        if (z != z).all():
            raise ValueError('p(x|z) forward contains NaN')

        return  x_con

class p_t_z(nn.Module):

    def __init__(self, dim_in=20, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of latent space z
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))

        out = bernoulli.Bernoulli(out_p)
        return out

class p_y_zt(nn.Module):

    def __init__(self, dim_in=20, nh=3, dim_h=20, dim_out=1, output_binary=False):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out
        self.output_binary = output_binary

        # Separated forwards for different t values

        self.input_t0 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t0 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t0 = nn.Linear(dim_h, dim_out)

        self.input_t1 = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden_t1 = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, z, t):
        # Separated forwards for different t values, TAR

        x_t0 = F.elu(self.input_t0(z))
        for i in range(self.nh):
            x_t0 = F.elu(self.hidden_t0[i](x_t0))
        mu_t0 = F.elu(self.mu_t0(x_t0))

        x_t1 = F.elu(self.input_t1(z))
        for i in range(self.nh):
            x_t1 = F.elu(self.hidden_t1[i](x_t1))
        mu_t1 = F.elu(self.mu_t1(x_t1))
        # set mu according to t value
        if self.output_binary:
            y = bernoulli.Bernoulli(torch.sigmoid(mu_t0*(1-t) + mu_t1*t))
        else:
            y = normal.Normal((1-t)*mu_t0 + t * mu_t1, 1)

        return y

####### Inference model / Encoder #######
class q_t_x(nn.Module):

    def __init__(self, dim_in=1, nh=1, dim_h=20, dim_out=1):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        self.output = nn.Linear(dim_h, dim_out)

    def forward(self, x):
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # for binary outputs:
        out_p = torch.sigmoid(self.output(x))
        out = bernoulli.Bernoulli(out_p)

        return out

class q_y_xt(nn.Module):

    def __init__(self, dim_in=1, nh=3, dim_h=20, dim_out=1, output_binary=False):
        super().__init__()
        # save required vars
        self.nh = nh
        self.dim_out = dim_out
        self.output_binary = output_binary

        # dim_in is dim of data x
        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])
        # separate outputs for different values of t
        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)

    def forward(self, x, t):
        # Unlike model network, shared parameters with separated heads
        x = F.elu(self.input(x))
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))
        # only output weights separated
        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        # set mu according to t, sigma set to 1
        if self.output_binary:
            y = bernoulli.Bernoulli(torch.sigmoid(mu_t0*(1-t) + mu_t1*t))
        else:
            y = normal.Normal((1-t)*mu_t0 + t * mu_t1, 1)
        return y

class q_z_tyx(nn.Module):

    def __init__(self, dim_in=2, nh=3, dim_h=20, dim_out=20):
        super().__init__()
        # dim in is dim of x + dim of y
        # dim_out is dim of latent space z
        # save required vars
        self.nh = nh

        # Shared layers with separated output layers

        self.input = nn.Linear(dim_in, dim_h)
        # loop through dimensions to create fully con. hidden layers, add params with ModuleList
        self.hidden = nn.ModuleList([nn.Linear(dim_h, dim_h) for _ in range(nh)])

        self.mu_t0 = nn.Linear(dim_h, dim_out)
        self.mu_t1 = nn.Linear(dim_h, dim_out)
        self.sigma_t0 = nn.Linear(dim_h, dim_out)
        self.sigma_t1 = nn.Linear(dim_h, dim_out)
        self.softplus = nn.Softplus()

    def forward(self, xy, t):
        # Shared layers with separated output layers
        # print('before first linear z_infer')
        # print(xy)
        x = F.elu(self.input(xy))
        # print('first linear z_infer')
        # print(x)
        for i in range(self.nh):
            x = F.elu(self.hidden[i](x))

        mu_t0 = self.mu_t0(x)
        mu_t1 = self.mu_t1(x)
        sigma_t0 = self.softplus(self.sigma_t0(x))
        sigma_t1 = self.softplus(self.sigma_t1(x))

        # Set mu and sigma according to t
        z = normal.Normal((1-t)*mu_t0 + t * mu_t1, (1-t)*sigma_t0 + t * sigma_t1)
        return z