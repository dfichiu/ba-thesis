from IPython.display import display, HTML, Markdown as md
import itertools

from lib.memory import DSDM
from lib.utils import inference, utils, preprocess

import math
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
import typing 

def generate_atomic_HVs_from_tokens_and_add_them_to_cleanup(
    dim: int,
    cleanup: dict,
    tokens: typing.List[str]
) -> None:
    """
    """
    for token in tokens:
        # Check if the token has been encountered before by querying the cleanup memory.
        entry = cleanup.get(token)
        # If it hasn't, 
        if entry == None:
            # Generate a random HV representation for the token.
            atomic_HV = thd.MAPTensor.random(1, dim)[0]
            # Add the HV to the cleanup memory.
            cleanup[token] = atomic_HV
    
    return


def generate_chunk_representations_and_save_them_to_memory(
    dim: int,
    cleanup: dict,
    memory: typing.List[DSDM.DSDM],
    tokens: typing.List[str],
    chunk_sizes: typing.List[int] = [],
    output: bool = False
) -> None:
    """
    
    """
    # "n" represents the no. of tokens in the sentence, which is also the max. no. of tokens 
    # that can be grouped to form a chunk.
    n = len(tokens)
    chunk_sizes = np.array(chunk_sizes, dtype=int)

    # Generate all possible chunks.
    if len(chunk_sizes) == 0:
        chunk_sizes = np.arange(1, n + 1)
    else:
        # Remove lengths which are bigger than the maximum chunk length.
        chunk_sizes = chunk_sizes[chunk_sizes <= n]
   
    for no_tokens in chunk_sizes:
        if output:
            print("no. of tokens: ", no_tokens)
        for i in range(n):
            if output:
                print("start index: ", i)
            # If there are not enough tokens left to construct a chunk comprised of "no_tokens", break. 
            if i + no_tokens > len(tokens):
                if output:
                    print("Not enough tokens left.")
                break 
            HC_representation = thd.MAPTensor.empty(1, dim)[0]

            # Construct HC representation.
            for j in range(no_tokens):
                if output:
                    print(tokens[i + j])
                HC_representation += cleanup[tokens[i + j]]

            # Save the chunk HC representation to memory.
            memory.save(HC_representation)

    return


def online_learning_with_inference(
    cleanup: dict,
    memories: typing.List[DSDM.DSDM],
    data_path: str,
    chunk_sizes: typing.List[int],
    epochs: typing.List[int],
    infer=False,
    inference_sentences=None,
    tracked_tokens=None,    
) -> typing.Union[tuple[pd.DataFrame, pd.DataFrame], None]:
    """
    """
    # Load data.
    text = utils.load_text(data_path)

    # Preprocess data. 
    sentences_tokens = preprocess.preprocess_text(text)
    
    
    if infer:
        sims_dfs = {} 
        tracked_tokens_sims_dfs = {}
        index = list(itertools.product(inference_sentences, tracked_tokens))
    
    # If all sentences should be trained for the same number of epochs.
    if len(epochs) == 1:
        epochs = len(sentences_tokens) * epochs
    

    for (idx, sentence_tokens), sentence_epochs in zip(enumerate(sentences_tokens), epochs):
        # Generate atomic HVs for unknown tokens.
        generate_atomic_HVs_from_tokens_and_add_them_to_cleanup(
            memories[list(memories.keys())[0]].address_size,
            cleanup,
            sentence_tokens
        )
        for epoch in range(sentence_epochs):
            if inference:
                column = 'similarity' + '_' + str(idx + 1) + '_' + str(epoch + 1)
            
            for memory_type, memory in memories.items():  # Memory loop
                # Learning: Construct the chunks of each sentence and save them to memory.
                generate_chunk_representations_and_save_them_to_memory(
                    memory.address_size,
                    cleanup,
                    memory,
                    sentence_tokens,
                    chunk_sizes=chunk_sizes
                )
            
                # Inference
                if infer:
                    # Get tabale with token similarities for each sentece.
                    sims_df = inference.infer(
                        memory.address_size,
                        cleanup,
                        memory,
                        inference_sentences
                    )
                    # Rename similarity column to reflect the epoch the similarity was obtained in.
                    sims_df = sims_df.rename(columns={'similarity': column})
                    
                    if epoch == 0:  # First epoch
                        sims_dfs[memory_type] = sims_df
                    if epoch == sentence_epochs - 1 and sentence_epochs - 1 > 0:  # Last epoch, which shouldn't be the first epoch because the column would get duplicated.
                        sims_dfs[memory_type] = sims_dfs[memory_type].merge(sims_df, left_index=True, right_index=True)

                    # Track similarities of partiuclar tokens over the entire learning scenaio.
                    if len(tracked_tokens_sims_dfs) < len(memories): 
                        tracked_tokens_sims_dfs[memory_type] = sims_df.loc[index]
                    else:
                        # Add similarities to global similarities table.
                        tracked_tokens_sims_dfs[memory_type][column] = sims_df.loc[index][column]
                        
    
    if infer: 
        return sims_dfs, tracked_tokens_sims_dfs
    else:
        return
