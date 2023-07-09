"""This file implements DSDM."""
from hashlib import sha256
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import random

import pandas as pd
import pathlib
from preprocess import preprocess_text

from sklearn.neighbors import LocalOutlierFactor

import torch
import torchhd as thd
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F 
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class that implements a self-organizing neural network which models a DSDM.
class DSDM(nn.Module):
    def __init__(self, Time_period, n_mini_batch, model):
        super(DSDM, self).__init__()
        self.dim_address = dim_address
        self.Time_period = Time_period 
        self.ema = 2 / (Time_period + 1)
        self.n_mini_batch = n_mini_batch
        self.count = 0
        self.T = 1
        self.A = torch.zeros(1, dim_address).to(device)
        self.p_norm = "fro"
        self.Error = torch.zeros(len(self.Address)).to(device)
        self.global_error = 0
        self.Time_period_Temperature = self.ema
        self.ema_Temperature = (2 / (self.Time_period_Temperature + 1))
        self.memory_global_error = torch.zeros(1)
        self.memory_min_distance = torch.zeros(1)
        self.memory_count_address = torch.zeros(1)
        self.dataset_name = "MNIST"
        
        self.acc_after_each_task = []
        self.acc_aft_all_task = []
        self.stock_feat = torch.tensor([]).to(device)
        self.forgetting = []
        self.N_prune = 5000 # Pruning threshold
        self.prune_mode = "balance"
        self.n_neighbors = 20
        self.contamination = "auto"
        self.pruning = False 
        self.cum_acc_activ = False
        self.batch_test = True
        
        self.reset()
        
    def reset(self):
        self.ema = 2 / (self.Time_period + 1)
        self.ema_Temperature = (2 / (self.Time_period_Temperature + 1))
        self.count = 0
        self.Address = torch.zeros(1, self.n_feat).to(device)
        self.M = torch.zeros(1, self.n_class).to(device)
        self.Error = torch.zeros(len(self.Address)).to(device)
        self.global_error = 0
        self.memory_global_error = torch.zeros(1)
        self.memory_min_distance = torch.zeros(1)
        self.memory_count_address = torch.zeros(1)
        
    def retrieve(self, query_address, batch_test=False):
        pass 
    
    def prune(self):
        pass
        
    def test(self, testloader):
        pass
    
    def test_idx(self, test_dataset_10_way_split, idx_test):
        pass
    
    def save(self, query_address, query_content, coef_global_error):
        """Add an item (target_address, target_content) to memory."""
        # Compute the address distances.
        address_distances = address_distance_function(query_address, self.A)
        # Get the minimum distance and the corresponding address index.  
        min_distance = torch.min(address_distances, dim=0)[0].item()
        # Adjust parameter based on the minimum distance..
        self.global_error += self.ema_Temperature * (min_distance - self.global_error)
        # Check if the minimum distance is bigger than the adaptive threshold.
        # If the minimum distance is bigger than the recursive temperature, add the item to memory.
        if min_distance >= self.global_error * coef_global_error:
            # Add a new entry to the address matrix/tensor equal to the query address.
            self.Address = torch.cat((self.Address, query_address.view(1, -1)))
            # If DSDM is used associatively, add a new entry to
            # the content matrix/tensor equal to the target content.
            if self.memory_type == "associative":
                self.M = torch.cat((self.M, query_content.view(1, -1)))
        # If the minimum distance is not bigger, then:
        else:
            # Apply the softmin function to the distance tensor the calculate the softmin weights.
            weights = F.softmin(address_distances/self.T, dim=-1)
            # Multiply the softmin weights by the learning rate to obtain the weights.
            weights *= self.ema 
            for i in range(len(self.Address)):
                # Update memory address.
                address_update_step(weights[i], query_address, self.A[i])
                if self.memory_type == "associative":
                    # Update memory content.
                    content_update_step(weights[i], query_address, self.C[i])
        
        return




