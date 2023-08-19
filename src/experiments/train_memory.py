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
from lib.utils import cleanup, inference, preprocess, sequence, utils

import networkx as nx
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import pickle
import string
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "bert-base-uncased"  # Has 12 layers.
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

MAXIMUM_SEQUENCE_LENGTH = 512

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
    parser.add_argument("--chunk_sizes", help="TODO")
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
        )
        return _cleanup, memory
    
    
def remove_duplicates(memory):
    global_keep_mask = torch.tensor([True] * len(memory.addresses)).to(device)
    
    for idx, address in enumerate(memory.addresses):
        if global_keep_mask[idx].item():
            cos = torch.nn.CosineSimilarity()
            keep_mask = cos(memory.addresses, address) < 0.95
            # Keep current address
            keep_mask[idx] = True
            global_keep_mask &= keep_mask

    if global_keep_mask.sum().item() > 0:
        # Remove similar addresses
        memory.addresses = memory.addresses[global_keep_mask]
        # Remove bins
        memory.scores = memory.scores[global_keep_mask]
        

def average_out_and_remove_rows(t: torch.tensor, averages_idx, remove_idx):
    for average_idx in averages_idx:  # The nested lists can have different dimensions.
        # Replace the attention scores of the first token with the average of the token attention scores.
        t[min(average_idx)] = torch.mean(t[average_idx], dim=0, keepdim=True)
    return t[~remove_idx]


def preprocess_attention_scores(attention_scores, averages_idx, remove_idx):
    attention_scores = average_out_and_remove_rows(attention_scores, averages_idx, remove_idx)
    attention_scores = attention_scores.transpose(0, 1)
    attention_scores = average_out_and_remove_rows(attention_scores, averages_idx, remove_idx)
    return attention_scores.transpose(0, 1)
    

def train_memory(cleanup, memory, args):
    train_idx = np.random.randint(0, len(wiki_dataset), size=args.train_size)
    
    n_documents = 0
    n_sentences = 0
    
    for i in tqdm(train_idx):
        n_documents = (n_documents + 1) % args.pruning_frequency
        text = wiki_dataset[int(i)]['text']
        memory.add_wiki_article(int(i))
        
        sentences = preprocess.split_text_into_sentences(text)
        
        for sentence in sentences:
            if args.pruning_frequency_type == "sentence":
                n_sentences = (n_sentences + 1) % args.pruning_frequency
            inputs = tokenizer(sentence, return_tensors="pt")
            if inputs['input_ids'].shape[1] > MAXIMUM_SEQUENCE_LENGTH:
                break
            
            outputs = model(**inputs, output_attentions=True)
            attention_matrix = outputs.attentions
            
            encoding = tokenizer.encode(sentence)
            labels = tokenizer.convert_ids_to_tokens(encoding)
    
            i = 0
            averages_idx = []
            while i < len(labels) - 1:
                j = i + 1
                average_idx = []
                while labels[j].startswith('#'):
                    average_idx.append(j)
                    labels[i] += labels[j].replace('#', '')
                    j += 1
                if average_idx != []:
                    average_idx.append(i)
                    averages_idx.append(average_idx)
                i = j
            
            hashtag_idx = np.array([label.startswith("#") for label in labels])
            stopwords_idx = np.array([label in stopwords.words('english') for label in labels])
            punctuation_idx = np.array([label in string.punctuation for label in labels])
            remove_idx = hashtag_idx | punctuation_idx | stopwords_idx
            labels = np.array(labels)[~remove_idx]
            labels = labels[1:(len(labels) - 1)]
    
            layer = 0
            
            for head in range(12):
                head_scores_raw_tensor = attention_matrix[layer][0][head].detach().clone()
                
                head_scores_raw_tensor = preprocess_attention_scores(head_scores_raw_tensor, averages_idx, remove_idx)
                
                head_scores_raw = head_scores_raw_tensor.cpu().detach().numpy()
                
                head_scores = head_scores_raw[1:(len(head_scores_raw) - 1), 1:(len(head_scores_raw) - 1)].copy()
            
                head_scores[head_scores < args.attention_score_threshold] = 0
                
                G = nx.from_numpy_array(head_scores, create_using=nx.DiGraph())
            
                n_tokens = len(labels)
                means, sequences = sequence.construct_sequences(G, n_tokens)
                    
                df = pd.DataFrame(data=[sequences, means]).T.rename(columns={0: 'seq',  1: 'score'})
                
                if len(df) > 0:
                    df['len'] = df['seq'].map(sum)
                    df['score'] = df['score'].astype('float64')
                    df = df.sort_values(by=['score', 'len'], ascending=[False, False]).reset_index(drop=True)
                    
                    # Select sequences to be save to memory.
                    if args.n_sequences is not None:
                        filtered_df = df.head(args.n_sequences)
                    elif args.chunk_score_threshold is not None:
                        filtered_df = df[df['score'] >= args.chunk_score_threshold]
                    else:
                        filtered_df = df.head(3)
                
                    # Save sequences to memory.
                    for i in range(len(filtered_df)):
                        memory.save(
                            inference.generate_query(
                                memory.address_size,
                                cleanup,
                                labels[filtered_df['seq'][i].astype(bool)]
                            ),
                            filtered_df['score'][i]
                        )
                        
            # Sentence-level pruning            
            if (
                args.prune_mode is not None
                and args.pruning_frequency_type == "sentence"
                and n_sentences % args.pruning_frequency == 0
            ):
                memory.prune()
                
        # Document-level pruning            
        if (
            args.prune_mode is not None
            and args.pruning_frequency_type == "document"
            and n_documents % args.pruning_frequency == 0
        ):
            memory.prune()
        
        if args.remove_duplicates and n_documents % 1000 == 0:
            remove_duplicates(memory)
            
    return

def save_memory(cleanup, memory):
    now = datetime.now()
    
    if not os.path.exists('memories'):
        os.makedirs('memories')
    if not os.path.exists('cleanups'):
        os.makedirs('cleanups')
        
    with open(f'memories/memory_{now}.pkl', 'wb') as outp:
        pickle.dump(memory, outp, pickle.HIGHEST_PROTOCOL)
    with open(f'cleanups/cleanup_{now}.pkl', 'wb') as outp:
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