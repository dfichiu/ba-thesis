"""This file implements DSDM."""
from IPython.display import display, Markdown as md
import ipywidgets as widgets
import itertools
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import random

import pandas as pd

from sklearn.metrics import pairwise_distances
from sklearn.neighbors import LocalOutlierFactor

import torch
import torchhd as thd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

# Type checking
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DSDM class
class DSDM(nn.Module):
    def __init__(self, address_size, ema_time_period, learning_rate_update, temperature, normalize=False):
        super(DSDM, self).__init__()
        self.address_size = address_size
        self.addresses = torch.tensor([]).to(device)

        self.normalize = normalize

        self.ema = 0
        self.ema_time_period = ema_time_period
        self.ema_temperature = 2 / (self.ema_time_period + 1)
        
        self.learning_rate_update = learning_rate_update

        self.temperature = temperature
        
        
    def retrieve(self, query_address):
        with torch.no_grad():
            retrieved_content = torch.tensor([]).to(device)

            cos = torch.nn.CosineSimilarity()
            # Calculate the cosine similarities.
            if self.normalize: 
                similarities = cos(self.addresses.sgn(), query_address.sgn())
            else:
                similarities = cos(self.addresses, query_address)
            # Cosine distance tensor
            distances = 1 - similarities

            # Calculate the softmin weights.
            softmin_weights = F.softmin(distances/self.temperature, dim=-1)

            # Weight the memory addresses with the softmin weights.
            weighted_addresses = torch.matmul(softmin_weights, self.addresses.to(device)).view(-1)

            # Pool the weighted memory addresses to create the output.
            retrieved_content = torch.sum(weighted_addresses.view(1, -1), 0)

        return retrieved_content   

    
    def save(self, query_address):
        # The memory is instantiated with the first observation.
        if self.addresses.shape[0] == 0:
            self.addresses = torch.cat((self.addresses, query_address.view(1, -1)))
            
            return
        
        cos = torch.nn.CosineSimilarity()
        # Calculate the cosine similarities.
        if self.normalize: 
            similarities = cos(self.addresses.sgn(), query_address.sgn())
        else:
            similarities = cos(self.addresses, query_address)

        # Calculate the cosine distances.
        distances = 1 - similarities
        # Get the minimum distance and the corresponding address index.  
        min_distance = torch.min(distances, dim=0)[0].item()
        
        # Calculate EMA for current chunk.
        self.ema += self.ema_temperature * (min_distance - self.ema)
        
        # Check if the minimum distance is bigger than the adaptive threshold.
        if min_distance > self.ema: # If the minimum distance is bigger, create a new address.
            # Add a new entry to the address matrix/tensor equal to the target address.
            self.addresses = torch.cat((self.addresses, query_address.view(1, -1)))
        else: # If the minimum distance is smaller or equal, update the memory addresses.
            # Apply the softmin function to the distance tensor the get the softmin weights.
            softmin_weights = F.softmin(distances/self.temperature, dim=-1)
            # Update the memory address space.
            self.addresses += self.learning_rate_update * torch.mul(softmin_weights.view(-1, 1), query_address - self.addresses)
         
        return

    # TODO: Original unadapted DSDM code.
    def prune(self):
        N_pruning = self.N_prune  # Maximum no. of (address) nodes the memory can have. 
        n_class = self.M.size(1)
        # If the maximum number of nodes has been reached, apply LOF
        # to get normalcy scores.
        if len(self.Address) > N_pruning:   
            clf = LocalOutlierFactor(n_neighbors=min(len(self.Address), self.n_neighbors), contamination=self.contamination)
            A = self.Address
            M = self.M
            y_pred = clf.fit_predict(A.cpu())
            X_scores = clf.negative_outlier_factor_
            x_scor = torch.tensor(X_scores)

            # "Naive" pruning mode.
            if self.prune_mode == "naive":
                if len(A) > N_pruning:
                    prun_N_addr = len(A) - N_pruning # No. of addresses that must be pruned out.
                    val, ind = torch.topk(x_scor, prun_N_addr) 
                    idx_remove = [True] * len(A)
                    for i in ind:
                        idx_remove[i] = False
                    self.M = self.M[idx_remove] # Delete content from address.
                    self.Address = self.Address[idx_remove] # Delete address.

            # "Balance" pruning mode.
            # Idea: Prune from each class instead of the nodes with the highest densities.
            if self.prune_mode == "balance":
                prun_N_addr = len(A) - N_pruning  # No. of addresses that must be pruned out.
                mean_addr = N_pruning // n_class  # Max. number of allowed nodes per class.
                val, ind = torch.sort(x_scor, descending=True)

                count = prun_N_addr
                idx_remove = [True] * len(A)
                idx = 0
                arg_m = torch.argmax(M, axis=1)  # Get predicted class.
                N_remaining = torch.bincount(arg_m)  # Count the frequency of each value, i.e., no. of predictions for each class.
                while count != 0:
                    idx +=1
                    indice = ind[idx]
                    if N_remaining[arg_m[indice]] > (N_pruning // n_class):
                        N_remaining[arg_m[indice]] -= 1
                        idx_remove[ind[idx]] = False
                        count-=1
                self.M = self.M[idx_remove]
                self.Address = self.Address[idx_remove]
        return

    def get_memory_type(self) -> str:
        """"""
        return "normalized" if self.normalize == True else "unnormalized"