import os
import torch
def make_dir(path):
    dirpath = os.path.dirname(path)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")