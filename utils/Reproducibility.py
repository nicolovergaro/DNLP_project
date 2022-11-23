import torch
import numpy
import random

SEED = 0

def make_it_reproducible(seed=SEED):
    random.seed(seed)
    numpy.random.seed(seed)
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)
    
def get_generator(seed=SEED):
    g = torch.Generator()
    g.manual_seed(0)
    return g
