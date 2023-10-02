"""This implements the class that stores the token-hypervector associations."""
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

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Cleanup():
    """
    The codebook used to save token-hypervector associations.
    
    Attributes:
        dim: Dimension of the hypervectors.
        n: Token/Row counter.
        index: Dictionary where the keys are the tokens and the values
          the row indices of the associatd hypervectors.
        items: Two-dimensional tensor storing the hypervectors.
    """
    
    def __init__(self, dim: int):
        self.dim = dim
        self.n = 0  # Token/Row counter.
        self.index = {}
        self.items = torch.tensor([]).to(device)

    def add(self, token: str):
        """
        Adds a token to the codebook.
        
        A token is added to codebook only if it hasn't
        been encountered before.
        """
        if self.index.get(token) is None:
            # Add token to the dictionary.
            self.index[token] = self.n
            self.n += 1
            
            # Generate a random MAP hypervector for the token.
            hv = thd.MAPTensor.random(1, self.dim)[0].to(device)
            self.items = torch.cat((self.items, hv.view(1, -1)))

    def get_item(self, token: str):
        """
        Retrieves the hypervector associated with a token.
        """
        # Get the row index of the correspondig HV.
        index = self.index.get(token)
        if index is not None:
            # Return the corresponding HV.
            return self.items[index]
        else:
            return None
