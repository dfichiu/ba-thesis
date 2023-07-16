from IPython.display import display, HTML, Markdown as md
import ipywidgets as widgets
import itertools

from lib.memory import DSDM
from lib.utils import preprocess

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import random

import pandas as pd
import pathlib

from sklearn.metrics import pairwise_distances

import torch
import torchhd as thd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

# Type checking
from typing import List

def column_output(
    memories: List[DSDM.DSDM],
    tables: dict,
    horizontal_output: bool = False
) -> None:
    """ """
    if horizontal_output:
        outs = [widgets.Output() for _ in range(len(memories))]

        for out, (memory_type, _) in zip(outs, memories.items()):
            with out:
                display(md(f"#### <ins>{memory_type.capitalize()}</ins>"))
                display(tables[memory_type])

        display(widgets.HBox(outs))
    else:
        for memory_type, _ in memories.items():
            display(md(f"#### <ins>{memory_type.capitalize()}</ins>"))
            display(tables[memory_type])
        
    return

def display_toc() -> None:
    display(md("## [Initial training](#Initial-training)"))
    display(md("### [   Extracted concepts](#initial-training-extracted-concepts)"))
    display(md("### [   Tracked tokens similarities](#initial-training-tracked-tokens-similarities)"))
    display(md("### [   Memory state](#initial-training-memory-state)"))
    
    display(md("## [Training](#training)"))
    display(md("### [   Extracted concepts](#training-extracted-concepts)"))
    display(md("### [   Tracked tokens similarities](#training-tracked-tokens-similarities)"))
    display(md("### [   Memory state](#training-memory-state)"))
    


def fix_seed(seed: int = 42) -> None:
    display(md(f"Using seed: {seed}"))

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def load_text(path:str) -> str:
    """Load text from file path."""
    file = open(path)
    return file.read()


def compute_distances_gpu(X, Y):
    """Compute Euclidean distance."""
    return torch.sqrt(-2 * torch.mm(X,Y.T) +
                    torch.sum(torch.pow(Y, 2),dim=1) +
                    torch.sum(torch.pow(X, 2),dim=1).view(-1,1))

