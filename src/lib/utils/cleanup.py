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


class Cleanup():
    def __init__(self, dim):
        self.dim = dim
        self.n = 0  # Token counter.
        self.index = {}
        self.items = torch.tensor([]).to(device)

    def add(self, token):
        if self.index.get(token) is None:
            self.index[token] = self.n
            self.n += 1
            
            hv = thd.MAPTensor.random(1, self.dim)[0].to(device)
            self.items = torch.cat((self.items, hv.view(1, -1)))

    def get_item(self, token):
        # Get index of correspondig HV.
        index = self.index.get(token)
        if index is not None:
            # Return the corresponding HV.
            return self.items[index]
        else:
            return None
