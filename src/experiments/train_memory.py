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

# Torch settings: Disable gradients.
torch.set_grad_enabled(False)

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_name = "bert-base-uncased"  # Has 12 layers.
# Load pre-trained Wordpiece tokenizer.
tokenizer = AutoTokenizer.from_pretrained(model_name)
# Load pre-trained BERT.
model = AutoModel.from_pretrained(model_name)

MAXIMUM_SEQUENCE_LENGTH = 512

# Load dataset.
wiki_dataset = datasets.load_dataset(
    "wikipedia",
    "20220301.en",
    cache_dir="/nfs/data/projects/daniela"
)['train']


def initialize_parser():
    """Define possible command line arguments."""
    parser=argparse.ArgumentParser()

    parser.add_argument("--memory_path", help="File path to saved memory")
    parser.add_argument("--address_size", type=int, help="Size of the VSA hypervectors")
    parser.add_argument("--ema_time_period", type=int, help="Number of days in EMA")
    parser.add_argument("--learning_rate_update", type=float, help="Learning rate used in the update step")
    parser.add_argument("--temperature", type=float, help="Base of the softmin exponent")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, help="Normalize addresses in cosine similarity computation")
#     parser.add_argument("--chunk_sizes", help="n-gram size")
    parser.add_argument("--prune_mode", help="Memory prune mode: 'fixed_size' and 'remove_percentage'")
    parser.add_argument("--max_size_address_space", type=int, help="If prune mode is set to 'fixed_size', the maximum size of the address space")
    parser.add_argument("--remove_percentage", type=float, help="If prune mode is set to 'remove_percentage', percentage of addresses to be remove")
    
    parser.add_argument("--train_size", type=int, help="Number of articles to train on")
    parser.add_argument("--attention_score_threshold", type=float, help="Attention score threshold")
    
    # Pruning settings
    parser.add_argument("--pruning_frequency_type", help="Pruning frequency type: document or sentence")
    parser.add_argument("--pruning_frequency", type=int, help="Pruning frequency")
    
    parser.add_argument("--safeguard_bins", action=argparse.BooleanOptionalAction, help="If True, do not remove addresses with a bin score lower than bin score threshold")
    parser.add_argument("--bin_score_threshold_type", help="Bin score threshold type: 'static' and 'dynamic' ")
    parser.add_argument("--bin_score_threshold", help="If bin score threshold type is static, bin score threshold")
    
    parser.add_argument("--safeguard_chunks", action=argparse.BooleanOptionalAction, help="Do not remove addresses with a chunk score lower than chunk score threshold")
    parser.add_argument("--chunk_score_threshold", type=float, help="Chunk score threshold")
    
    parser.add_argument("--n_sequences", type=int, help="Number of subsequences/self-attention head to be committed to memory")
    
#     parser.add_argument("--remove_duplicates", action=argparse.BooleanOptionalAction, help="")
    
    parser.add_argument("--remove_stopwords", action=argparse.BooleanOptionalAction, help="Remove stop words")

    return parser
    

def initialize_memory(args):
    """Initialize memory and associated codebook."""
    if args.memory_path is not None:
        # Continue training:
        # Load memory and codeboook from file.
        # TODO
        pass
    else:
        # Initialize new cleanup.
        _cleanup = cleanup.Cleanup(args.address_size)
        
        # Initialize new memory.
        memory = DSDM.DSDM(
            address_size=1000, #args.address_size,
            ema_time_period=100000, #args.ema_time_period,
            learning_rate_update=0, #args.learning_rate_update,
            temperature=args.temperature,
            normalize=False, #args.normalize,
            prune_mode=args.prune_mode,
            pruning_frequency_type=args.pruning_frequency_type,
            pruning_frequency=args.pruning_frequency,
            max_size_address_space=args.max_size_address_space,
            remove_percentage=args.remove_percentage,
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
        

def average_out_and_remove_rows(
    t: torch.tensor,
    averages_idx: list,
    remove_idx: np.array
) -> torch.Tensor:
    for average_idx in averages_idx:  # The nested lists can have different dimensions.
        # Replace the attention scores of the first token (subword)
        # with the average of the tokens' (subwords') attention scores.
        t[min(average_idx)] = torch.mean(t[average_idx], dim=0, keepdim=True)
    return t[~remove_idx] # Return matrix with removed entries.


def preprocess_attention_scores(
    attention_scores: torch.Tensor,
    averages_idx: list,
    remove_idx: np.array
) -> torch.Tensor:
    """
    Preprocess self-attention matrix.
    
    Average out rows associated with subwords to create entries of reconstructed
    words. Remove punctuation, stop words, and subwords. Apply same procedure to columns by
    transposing the matrix.
    """
    # Remove entries from rows.
    attention_scores = average_out_and_remove_rows(attention_scores, averages_idx, remove_idx)
    # Transpose matrix.
    attention_scores = attention_scores.transpose(0, 1)
    # Remove entries from columns.
    attention_scores = average_out_and_remove_rows(attention_scores, averages_idx, remove_idx)
    # Transpose matrix.
    return attention_scores.transpose(0, 1)
    

def train_memory(cleanup, memory, args):
    train_idx = np.random.randint(0, len(wiki_dataset) - 1000, size=1000000)
    # Select train articles.
    train_idx = train_idx[:args.train_size]
    # Manually add the articles from which the in-set inference sentences were selected.
    train_idx = np.append(np.array([6458629, 6458633, 6458645, 6458648, 6458659, 6458664, 6458665,
       6458667, 6458668, 6458573]), train_idx)
    
    n_documents = 0
    n_sentences = 0
    
    for i in tqdm(train_idx):
        if args.pruning_frequency_type == "document":
            n_documents = (n_documents + 1) % args.pruning_frequency
        text = wiki_dataset[int(i)]['text']
        memory.add_wiki_article(int(i))
        
        # Split text into sentences.
        sentences = preprocess.split_text_into_sentences(text)
        
        # Process each sentence sepparately.
        for sentence in sentences:
            if args.pruning_frequency_type == "sentence":
                n_sentences = (n_sentences + 1) % args.pruning_frequency
            # Get BERT input.
            inputs = tokenizer(sentence, return_tensors="pt")
            if inputs['input_ids'].shape[1] > MAXIMUM_SEQUENCE_LENGTH:
                # If the sentence is longer than the maximum no. of allowed tokens, skip it.
                break
            
            # Pass input through BERT.
            outputs = model(**inputs, output_attentions=True)
            attention_matrix = outputs.attentions
            
            encoding = tokenizer.encode(sentence)
            # Get sentence tokens.
            labels = tokenizer.convert_ids_to_tokens(encoding)
            
            # Note: Starting with the second subword,
            # subwords start with '##.'
            # Get indices of subwords for averaging and reconstruct
            # words from subwords. Each reconstructed word is saved
            # in the place of the first subword.
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
            
            # Construct multiple masks to indentify uninformative tokens:
            ## i) subwords: Start with '##;'
            ## ii) punctuation: Use string.punctuation to identify them;
            ## iii) other: Uninformative characters that are not part of 'string.punctuation;'
            ## iv) stop words: Use 'stopwords' from 'nltk.corpus.'
            # Then apply OR to construct global mask of uninformative tokens.
            hashtag_idx = np.array([label.startswith("#") for label in labels])
            stopwords_idx = np.array([label in stopwords.words('english') for label in labels])
            other_idx = np.array([(len(label) == 1 and (ord(label) == 8211 or ord(label) == 65288)) for label in labels])
            punctuation_idx = np.array([label in string.punctuation for label in labels])
            remove_idx = hashtag_idx | punctuation_idx | other_idx  
            if args.remove_stopwords:
                remove_idx |= stopwords_idx
            # Remove uninformative tokens from sentence
            # by applying global mask.
            labels = np.array(labels)[~remove_idx]
            # Remove '[CLS]' and '[SEP]' tokens from sentence tokens.
            labels = labels[1:(len(labels) - 1)]
    
            #layer = 0
            for layer in range(12):
                for head in range(12):
                    head_scores_raw_tensor = attention_matrix[layer][0][head].clone()#.detach().clone()
                    
                    # Remove self-attention matrix entries (rows & columns) of uninformative tokens.
                    head_scores_raw_tensor = preprocess_attention_scores(head_scores_raw_tensor, averages_idx, remove_idx)

                    head_scores_raw = head_scores_raw_tensor.numpy()#.cpu().detach().numpy()
                    
                    # Remove entries (rows & columns) associated with '[CLS]' and '[SEP]' tokens.
                    head_scores = head_scores_raw[1:(len(head_scores_raw) - 1), 1:(len(head_scores_raw) - 1)].copy()
                    
                    # Zero out entries with an attention weight
                    # lower than the attention score threshold.
                    head_scores[head_scores < args.attention_score_threshold] = 0
                    
                    # Construct weighted directed graph from matrix.
                    G = nx.from_numpy_array(head_scores, create_using=nx.DiGraph())
                    
                    # Construct subsequences and calculate associated
                    # chunk scores (i.e., averages of the associated attention weights).
                    # ----
                    # sequences: binary vector where the 
                    # 1-components indicate the tokens that are part of the subsequence;
                    # means: float vector with the chunk scores.
                    n_tokens = len(labels)
                    means, sequences = sequence.construct_sequences(G, n_tokens)

                    # Construct dataframe from subsequences.
                    df = pd.DataFrame(data=[sequences, means]).T.rename(columns={0: 'seq',  1: 'score'})

                    if len(df) > 0:
                        # Get subsequence length.
                        df['len'] = df['seq'].map(sum)
                        df['score'] = df['score'].astype('float64')
                        # Sort subsequences.
                        df = df.sort_values(by=['score', 'len'], ascending=[False, False]).reset_index(drop=True)

                        # Select sequences to be save to memory.
                        if args.n_sequences is not None:
                            filtered_df = df.head(args.n_sequences)
                        elif args.chunk_score_threshold is not None:
                            filtered_df = df[df['score'] >= args.chunk_score_threshold]
                        else:
                            # Fall-back value
                            filtered_df = df.head(1)

                        # Save sequences along chunk scores to memory.
                        for i in range(len(filtered_df)):
                            # Call 'generate_query' to construct token superposition.
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
        
#         if args.remove_duplicates and n_documents % 1000 == 0:
#             remove_duplicates(memory)
            
    return


def save_memory(cleanup, memory):
    now = str(datetime.now()).replace(':', "-").replace('.', '-')
    
    if not os.path.exists('memories/method2'):
        os.makedirs('memories/method2')
    if not os.path.exists('cleanups/method2'):
        os.makedirs('cleanups/method2')
        
    with open(f'memories/method2/memory_{now}.pkl', 'wb') as outp:
        pickle.dump(memory, outp, pickle.HIGHEST_PROTOCOL)
    with open(f'cleanups/method2/cleanup_{now}.pkl', 'wb') as outp:
        pickle.dump(cleanup, outp, pickle.HIGHEST_PROTOCOL)
        
    return
    

def main():
    # Set seeds.
    utils.fix_seed(41)

    # Initialize parser.
    parser = initialize_parser()

    # Parse command line arguemnts.
    args = parser.parse_args()

    # Intialize memory and cleanup.
    cleanup, memory = initialize_memory(args)

    # Train memory.
    train_memory(cleanup, memory, args)
    
    # Save codebook and memory to file.
    save_memory(cleanup, memory)

    return


if __name__ == "__main__":
    main()