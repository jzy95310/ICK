# Original post: https://github.com/thuizhou/Collaborating-Networks
# Author: Tianhui Zhou
# ArXiv: https://arxiv.org/abs/2110.01664
# Reference: Zhou, Tianhui, William E. Carson IV, and David Carlson. "Estimating potential outcome distributions 
# with collaborating causal networks." arXiv preprint arXiv:2110.01664 (2021).

import torch
from torch import nn
import torch.nn.functional as F

#estimate CDF
class cn_g(nn.Module):
    def __init__(self, input_dim):
        super(cn_g, self).__init__()
        self.k1 = 100
        self.k2 = 80
        self.fc1 = nn.Linear(input_dim+1, self.k1)
        self.fc2 = nn.Linear(self.k1, self.k2)
        self.fc3 = nn.Linear(self.k2, 1)

    def forward(self, y, x):
        data = torch.cat([y,x],dim=1)
        h1 = self.fc1(data)
        h1 = F.elu(h1)
        h2 = self.fc2(h1)
        h2 = F.elu(h2)
        h3 = self.fc3(h2)
        g_logit = h3
        return g_logit

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.normal_(m.weight, 0.0, .001)