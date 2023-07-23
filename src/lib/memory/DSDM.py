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

import torch
import torchhd as thd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

# Type checking
from typing import List

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DSDM(nn.Module):
    def __init__(
        self,
        address_size,
        ema_time_period,
        learning_rate_update,
        temperature,
        prune_mode=None,
        max_size_address_space=None,
        remove_percentage=None,
        bin_threshold=None,
        normalize=False
    ):
        super(DSDM, self).__init__()
        self.address_size = address_size
        self.addresses = torch.tensor([]).to(device)
        self.bins = torch.tensor([]).to(device)

        self.normalize = normalize

        self.ema = 0
        self.ema_time_period = ema_time_period
        self.ema_temperature = 2 / (self.ema_time_period + 1)
        
        self.learning_rate_update = learning_rate_update

        self.temperature = temperature

        # Set statistics counters.
        self.n_updates = 0
        self.n_expansions = 0
        self.n_deletions = 0

        # Set pruning hyperparameters.
        self.prune_mode = prune_mode
        self.max_size_address_space = max_size_address_space
        self.remove_percentage = remove_percentage
        self.bin_threshold = bin_threshold

    
    def get_memory_type(self) -> str:
        return "normalized" if self.normalize == True else "unnormalized"

    
    def set_temperature(self, temperature):
        self.temperature = temperature

    
    def set_learning_rate_update(self, learing_rate_update):
        self.learning_rate_update = learning_rate_update
        
        
    def retrieve(
        self,
        query_address,
        retrieve_mode="pooling",
        k=None
    
    ):
        with torch.no_grad():
            query_address = query_address.to(device)
            #retrieved_content = torch.tensor([]).to(device)
            self.prune()

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

            if retrieve_mode == "pooling":
                # Weight the memory addresses with the softmin weights.
                weighted_addresses = torch.matmul(softmin_weights, self.addresses.to(device)).view(-1)
    
                # Pool the weighted memory addresses to create the output and return it.
                return torch.sum(weighted_addresses.view(1, -1), 0)
            else:  # retrieve_mode == "top_k"
                return_mask = [False] * len(self.addresses)
                
                val, idx = torch.topk(
                    softmin_weights.view(1, -1),
                    k=k,
                    largest=True
                )

                # Convert tensor to flattened numpy array.
                idx = idx.cpu().detach().numpy().flatten()
                for i in idx:
                    return_mask[i] = True
                
                return self.addresses[return_mask]  

    
    def save(self, query_address):
        # The memory is instantiated with the first observation.
        query_address = query_address.to(device)
        
        if self.addresses.shape[0] == 0:
            self.addresses = torch.cat(
                (
                    self.addresses,
                    query_address.view(1, -1)
                )
            )
            self.bins = torch.cat(
                (
                    self.bins,
                    torch.tensor([0]).to(device)
                )
            )
            self.n_expansions += 1  
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
            self.addresses = torch.cat(
                (
                    self.addresses,
                    query_address.view(1, -1)
                )
            )
            self.bins = torch.cat(
                (
                    self.bins,
                    torch.tensor([0]).to(device)
                )
            )
            self.n_expansions += 1  
        else: # If the minimum distance is smaller or equal, update the memory addresses.
            # Apply the softmin function to the distance tensor the get the softmin weights.
            softmin_weights = F.softmin(distances/self.temperature, dim=-1)
            # Update the memory address space.
            self.addresses += self.learning_rate_update * torch.mul(softmin_weights.view(-1, 1), query_address - self.addresses)
            self.bins += softmin_weights
            self.n_updates += 1
            
        return

    
    def prune(self):
        keep_mask = [True] * len(self.addresses)  # Assume no pruning is needed.

        if self.prune_mode is not None: 
            if (self.prune_mode == "fixed-size" or self.prune_mode == "remove-percentage") and (len(self.addresses) > self.max_size_address_space):
                if self.prune_mode == "fixed-size":
                    n_prune = len(self.addresses) - self.max_size_address_space
                else:
                    n_prune = int(self.remove_percentage * len(self.addresses))
                    
                val, idx = torch.topk(
                    self.bins.view(1, -1),
                    k=n_prune,
                    largest=False
                ) 

                # Convert tensor to flattened numpy array.
                idx = idx.cpu().detach().numpy().flatten()
                for i in idx:
                    keep_mask[i] = False
                
            if self.prune_mode == "threshold":
                keep_mask = self.bins.view(1, -1) >= self.bin_threshold

        
        self.n_deletions += np.sum(keep_mask)
        # Prune memory space.
        self.addresses = self.addresses[keep_mask]  # Delete addresses.
        self.bins = self.bins[keep_mask]  # Delete bins.
            
        return

    
