#!/usr/bin/env python

import os
import sys

# Get the absolute path of the parent directory.
parent_dir = os.path.abspath(os.path.join(os.path.dirname("__file__"), ".."))

# Add the parent directory to the system path to be able to import modules from 'lib.'
sys.path.append(parent_dir)

import argparse
import datasets
from datetime import datetime

from lib.memory import DSDM
from lib.utils import cleanup, inference, learning, preprocess, sequence, utils

import networkx as nx
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pickle
import string
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# Torch settings: Disable gradient.
torch.set_grad_enabled(False)

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Duplicates
dups_found = 0

# model_name = "bert-base-uncased"  # Has 12 layers.
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = AutoModel.from_pretrained(model_name)

# MAXIMUM_SEQUENCE_LENGTH = 512

wiki_dataset = datasets.load_dataset(
    "wikipedia",
    "20220301.en",
    cache_dir="/nfs/data/projects/daniela"
)['train']


def initialize_parser():
    parser=argparse.ArgumentParser()

    parser.add_argument("--memory_path", help="Size of the VSA hypervectors.")
    parser.add_argument("--address_size", type=int, help="Size of the VSA hypervectors.")
    parser.add_argument("--ema_time_period", type=int, help="TODO")
    parser.add_argument("--learning_rate_update", type=float, help="TODO")
    parser.add_argument("--temperature", type=float, help="TODO")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="TODO")
    parser.add_argument("--chunk_size", type=int, help="TODO")
    parser.add_argument("--prune_mode", help="TODO")
    parser.add_argument("--max_size_address_space", type=int, help="TODO")
    parser.add_argument("--train_size", type=int, help="TODO")
    parser.add_argument("--attention_score_threshold", type=float, help="TODO")
    
    # Pruning settings
    parser.add_argument("--pruning_frequency_type", help="TODO")
    parser.add_argument("--pruning_frequency", type=int, help="TODO")
    
    parser.add_argument("--safeguard_bins", action=argparse.BooleanOptionalAction, help="TODO")
    parser.add_argument("--bin_score_threshold_type", help="TODO")
    parser.add_argument("--bin_score_threshold", help="TODO")
    
    parser.add_argument("--safeguard_chunks", action=argparse.BooleanOptionalAction, help="TODO")
    parser.add_argument("--chunk_score_threshold", type=float, help="TODO")
    
    parser.add_argument("--n_sequences", type=int, help="TODO")
    
    parser.add_argument("--remove_duplicates", action=argparse.BooleanOptionalAction, help="TODO")
    parser.add_argument("--duplicates_threshold", type=float, help="TODO")

    return parser
    

def initialize_memory(args):
    if args.memory_path is not None:
        # Load memory and cleanup dictionary from file path.
        # TODO
        pass
    else:
        # Initialize new cleanup.
        _cleanup = cleanup.Cleanup(args.address_size)
        
        # Initialize new memory.
        memory = DSDM.DSDM(
            address_size=args.address_size,
            ema_time_period=args.ema_time_period,
            learning_rate_update=args.learning_rate_update,
            temperature=args.temperature,
            normalize=args.normalize,
            prune_mode=args.prune_mode,
            max_size_address_space=args.max_size_address_space,
            safeguard_chunks=args.safeguard_chunks,
            chunk_score_threshold=args.chunk_score_threshold,
            safeguard_bins=args.safeguard_bins,
            bin_score_threshold_type=args.bin_score_threshold_type,
            bin_score_threshold=args.bin_score_threshold,
            chunk_size=args.chunk_size,
        )
        return _cleanup, memory
    
    
def remove_duplicates(memory, args):
    global dups_found
    global_keep_mask = torch.tensor([True] * len(memory.addresses)).to(device)
    
    for idx, address in enumerate(memory.addresses):
        if global_keep_mask[idx].item():
            cos = torch.nn.CosineSimilarity()
            keep_mask = cos(memory.addresses, address) < args.duplicates_threshold
            # Keep current address
            keep_mask[idx] = True
            global_keep_mask &= keep_mask

    if global_keep_mask.sum().item() > 0:
        dups_found += len(global_keep_mask) - global_keep_mask.sum().item()
        # Remove similar addresses
        memory.addresses = memory.addresses[global_keep_mask]
        # Remove bins
        memory.scores = memory.scores[global_keep_mask]
        

# def average_out_and_remove_rows(t: torch.tensor, averages_idx, remove_idx):
#     for average_idx in averages_idx:  # The nested lists can have different dimensions.
#         # Replace the attention scores of the first token with the average of the token attention scores.
#         t[min(average_idx)] = torch.mean(t[average_idx], dim=0, keepdim=True)
#     return t[~remove_idx]


# def preprocess_attention_scores(attention_scores, averages_idx, remove_idx):
#     attention_scores = average_out_and_remove_rows(attention_scores, averages_idx, remove_idx)
#     attention_scores = attention_scores.transpose(0, 1)
#     attention_scores = average_out_and_remove_rows(attention_scores, averages_idx, remove_idx)
#     return attention_scores.transpose(0, 1)
    

def train_memory(cleanup, memory, args):
    train_idx = np.random.randint(0, len(wiki_dataset) - 1000, size=50000)
    train_idx = train_idx[:args.train_size]
    train_idx = np.append(np.array([6458629, 6458633, 6458645, 6458648, 6458659, 6458664, 6458665,
       6458667, 6458668, 6458573]), train_idx)
    
    n_documents = 0
    n_sentences = 0

    # Article loop
    for i in tqdm(train_idx):
        memory.add_wiki_article(int(i))
        text = wiki_dataset[int(i)]['text']

        # Preprocess data. 
        sentences_tokens = preprocess.preprocess_text(text)
        
        # Sentence loop
        for sentence_tokens in sentences_tokens:
            # Generate atomic HVs for unknown tokens.
            learning.generate_atomic_HVs_from_tokens_and_add_them_to_cleanup(
                memory.address_size,
                cleanup,
                sentence_tokens
            )

            # Learning: Construct the chunks of each sentence and save them to memory.
            learning.generate_chunk_representations_and_save_them_to_memory(
                memory.address_size,
                cleanup,
                memory,
                sentence_tokens,
                chunk_sizes=[args.chunk_size]
            )
            
            # Sentence-level pruning            
            if (
                args.prune_mode is not None
                and args.pruning_frequency_type == "sentence"
                and n_sentences % args.pruning_frequency == 0
            ):
                memory.prune()
        
        
    if args.remove_duplicates:
        remove_duplicates(memory, args)
        
        global dups_found
        memory.removed_duplicates = dups_found
            
    return


def save_memory(cleanup, memory):
    now = datetime.now()
    
    if not os.path.exists('memories/method1'):
        os.makedirs('memories/method1')
    if not os.path.exists('cleanups/method1'):
        os.makedirs('cleanups/method1')
        
    with open(f'memories/method1/memory_{now}.pkl', 'wb') as outp:
        pickle.dump(memory, outp, pickle.HIGHEST_PROTOCOL)
    with open(f'cleanups/method1/cleanup_{now}.pkl', 'wb') as outp:
        pickle.dump(cleanup, outp, pickle.HIGHEST_PROTOCOL)
        
    return
    

def main():
    # Fix seed.
    utils.fix_seed(41)

    # Initialize parser.
    parser = initialize_parser()

    # Parse passed arguemnts.
    args = parser.parse_args()

    # Intialize memory and cleanup.
    cleanup, memory = initialize_memory(args)

    # Train memory.
    train_memory(cleanup, memory, args)
    
    # Save memory to file.
    save_memory(cleanup, memory)

    return


if __name__ == "__main__":
    main()