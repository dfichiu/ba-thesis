"""This file implements functions used during inference."""
from IPython.display import display, HTML, Markdown as md
import ipywidgets as widgets
import itertools

from lib.memory import DSDM
from lib.utils import cleanup, preprocess, utils

import math
import matplotlib
import matplotlib.pyplot as plt

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

import numpy
import numpy as np
import random

import pandas as pd
import pathlib

import seaborn as sns
import string

from sklearn.metrics import pairwise_distances

from transformers import AutoTokenizer, AutoModel

import torch
import torchhd as thd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

# Torch settings: Disable gradient.
torch.set_grad_enabled(False)

# Type checking
import typing

# Set device.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_query(
    dim: int,
    cleanup: cleanup.Cleanup,
    tokens: typing.List[str]
) -> torch.Tensor:
    """
    Computers and returns the superposition of a list of tokens.
    """
    n = len(tokens)
    # Empty hypvervector used to sum up the token hypervectors.
    hc_representation = thd.MAPTensor.empty(1, dim).to(device)

    # Iterate through all tokens.
    for i in range(n):
        # The token hasn't been encountered before.
        if cleanup.get_item(tokens[i]) == None:
            # Add the new token to the codebook.
            cleanup.add(tokens[i])
        # Add token-associated hypervector to superposition.
        hc_representation += cleanup.get_item(tokens[i])

    return hc_representation


def get_similarities_to_atomic_set(
    content: torch.Tensor,
    cleanup: cleanup.Cleanup,
    k: int = 10
) -> pd.DataFrame:
    """
    Recovers atomic hypervectors present in a superposition.
    
    Returns a dataframe consisting of the tokens associated
    with the atomic hypervectors and the cosine similarities between
    the superposition and the atomic hypervectors. The dataframe is sorted
    in descending order by the cosine similarity and only the top k 
    rows are returned.
    
    Args:
        content: Tensor representing superposition.
        cleanup: Codebook containing the atomic set information.
        k: Number of returned atomic hypervectors.
    """
    # Compute cosine similarities between superpoisition and atomic set.
    # Note: By storing the atomic set in a tensor,
    # the similarities can be computed using matrix multiplication
    # instead of iteratively.
    atomic_similarities = F.cosine_similarity(
        cleanup.items,
        content
    )
    # Get the k hypervectors with the highest cosine similarity.
    val, idx = torch.topk(
        atomic_similarities.view(1, -1),
        k=k,
        largest=True
    )
    # Construct dataframe.
    sims_df = pd.DataFrame(
        data={
            'token': np.array(list(cleanup.index))[idx.cpu().detach().numpy().flatten()],
            'similarity': np.round(val.cpu().detach().numpy().flatten(), 2)
        }
    )
    return sims_df


def get_most_similar_HVs(
    sims_df: pd.DataFrame,
    delta_threshold: float = 0.15
) -> str:
    """
    Joins the tokens with the highest cosine similarity.
    
    It was used in the initial normalization experiments. 
    (See src/normalization-experiments.)
    The function was used in the context of the sliding window
    n-gram method, where the similarities changed due to
    changes to the weights of the superpositions, to
    extract the concept an address represented. This is why
    the string is constructed from tokens between which the delta
    in similarity is lower than or equal to a threshold.
    
    Args:
        sims_df: DataFrame consisting of tokens and cosine similarities.
        delta_threshold: Threshold between consecutive similarities.
    """
    
    # Sort values: This is needed since similarity_next makes sense only in the context of a sorted df.
    df = sims_df.sort_values('similarity', ascending=False).reset_index(drop=True).copy()
    # Add column with the previous token's similarity.
    df['previous_token_similarity'] = df['similarity'].shift(1).values
    # Compute the differece between the similarities. 
    df['delta'] = df['previous_token_similarity'] - df['similarity']
    # Set the NaN value of the delta to '0', since the first token doesn't have a previous token.
    df['delta'] = df['delta'].fillna(0)
    # Get index of the first element whose delta is bigger than delta_threshold.
    # TODO: Consider - This might have the edge case of all the deltas decreasing by delta_threshold.
    unsimilar_df = df[df['delta'] > delta_threshold].head(1)
    # We initially assume that all the tokens are equally represented.
    idx_cut_in = len(unsimilar_df)

    if len(unsimilar_df) > 0:
        idx_cut_in = df[df['delta'] > delta_threshold].head(1).index[0]

    # Subdataframe with only the most similar tokens.
    most_similar_tokens_df = df.head(idx_cut_in)
    
    # Get concept as a string.
    concept = most_similar_tokens_df['token'].values
    display(concept)
    concept.sort()
    #print(concept)
    #display(df)
    return concept 
    

def display_and_get_memory_addresses(
    memory: DSDM.DSDM,
    cleanup: dict,
    k: int = 10,
    display_addresses: bool = False,
):
    """
    Displays the memory address space.
    
    It was used in the initial normalization experiments. 
    (See src/normalization-experiments.)
    """
    display(md(f"Number of addresses: **{len(memory.addresses)}**"))

    concepts_df = pd.DataFrame(columns=['memory_address', 'memory_concept'])
    
    for address in memory.addresses[: k]:
        sims_df = pd.DataFrame(columns=['token', 'similarity'])
        for key, item in cleanup.items():
            sims_df = pd.concat([sims_df, pd.DataFrame([{'token': key, 'similarity': thd.cosine_similarity(item,  address).item()}])])
        
        if display_addresses:
            display(sims_df.sort_values('similarity', ascending=False).reset_index(drop=True))
        concept = get_most_similar_HVs(sims_df)
        concepts_df = pd.concat([concepts_df, pd.DataFrame([{'memory_address': address, 'memory_concept': concept}])])
    
    concepts_df = concepts_df.reset_index(drop=True)
    concepts_df['memory_concept_str'] = concepts_df['memory_concept'].apply(lambda concept_list: " ".join(concept_list))
    #display(concepts_df)
    #display(sims_df.sort_values('similarity', ascending=False).reset_index(drop=True))
    #display(concepts_df)
    
    concepts_stats = pd.DataFrame(concepts_df.groupby('memory_concept_str').size()).rename(columns={0: 'count', 'memory_concept_str': 'concept'}).sort_values('count', ascending=False)
    # Rename index.
    concepts_stats.index.name = 'concept'
    #concepts_stats = concepts_stats.sort_values('concept', ascending=True)
    display(concepts_stats)
    
    return concepts_df



def bert_preprocessing(sentence: str, remove_stopwords: bool = True) -> typing.List[str]:
    """
    Preprocesses the sentence token list resulting from BERT's tokenizer.
    
    Passes a sentence through BERT's tokenizer, reconstructs words from
    subwords, and removes uninformative tokens.
    
    The uniformative tokens are:
       i) subwords: Start with '##;'
       ii) punctuation: Use string.punctuation to identify them;
       iii) other: Uninformative characters that are not part of 'string.punctuation;'
    Optionally, stop words ('stopwords' from 'nltk.corpus') are also removed.
       
    Args:
        sentence: Sentence whose token list is processes.
        remove_stopwords: If True, stop words are also removed.
    """
    model_name = "bert-base-uncased"  # Has 12 layers
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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
    dash_idx = np.array([(len(label) == 1 and ord(label) == 8211) for label in labels])
    remove_idx = hashtag_idx | punctuation_idx | dash_idx
    if remove_stopwords:
        remove_idx |= stopwords_idx
    labels = np.array(labels)[~remove_idx]
    return labels[1:(len(labels) - 1)]


def infer(
    dim: int,
    cleanup: cleanup.Cleanup,
    memory: DSDM.DSDM,
    inference_sentences: typing.List[typing.Union[str, typing.List]],
    retrieve_mode: str = "pooling",
    k=None,
    output=False
) -> typing.Union[pd.DataFrame, typing.List[typing.List]]:

    retrieved_contents = []
        
    for s in inference_sentences:
        #tokens_list = preprocess.preprocess_text(s)[0] if isinstance(s, str) else s
        tokens_list = bert_preprocessing(s)
        
        retrieved_content = memory.retrieve(
            query_address=generate_query(
                dim,
                cleanup,
                tokens_list
            ),
            retrieve_mode=retrieve_mode,
            k=k,
        )
        retrieved_contents.append(retrieved_content)

    return retrieved_contents



def get_similarity_matrix_of_addresses_mapping_to_same_concepts(concepts_df: dict) -> None:
    tmp_df = pd.DataFrame(concepts_df.groupby('memory_concept_str')['memory_address'].apply(list)).reset_index()
    for i in range(len(tmp_df)):
        address_list = tmp_df['memory_address'][i]
        concept = tmp_df['memory_concept_str'][i]
        
        if len(address_list) > 1: 
            stacked_tensor = torch.stack(address_list, dim=0)
            pairwise_similarities = torch.nn.functional.cosine_similarity(
                stacked_tensor.unsqueeze(1),
                stacked_tensor.unsqueeze(0),
                dim=2
            )
            similarity_matrix_np = pairwise_similarities.numpy()

            # Display the similarity matrix as a plot.
            sns.heatmap(similarity_matrix_np, annot=True, cmap="YlGnBu")
            plt.title(f"Concept '{concept}' ")
            plt.show()
            
    return