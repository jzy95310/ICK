import torch

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
OPTIMIZERS = {
    'adam': torch.optim.Adam, 
    'sgd': torch.optim.SGD, 
    'rmsprop': torch.optim.RMSprop, 
    'adagrad': torch.optim.Adagrad, 
    'adadelta': torch.optim.Adadelta, 
    'adamw': torch.optim.AdamW,
    'sparseadam': torch.optim.SparseAdam,
    'adamax': torch.optim.Adamax,
    'asgd': torch.optim.ASGD,
    'lbfgs': torch.optim.LBFGS,
    'rprop': torch.optim.Rprop,
}