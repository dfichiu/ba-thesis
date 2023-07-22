from IPython.display import display, HTML, Markdown as md
import ipywidgets as widgets
import itertools

from lib.memory import DSDM
from lib.utils import preprocess, utils

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np
import random

import pandas as pd
import pathlib

import seaborn as sns

from sklearn.metrics import pairwise_distances

import torch
import torchhd as thd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F 

# Type checking
import typing


def generate_query(
    dim: int,
    cleanup: dict,
    tokens: typing.List[str]
) -> torch.Tensor:
    """
    """
    n = len(tokens)
    HC_representation = thd.MAPTensor.empty(1, dim)

    # Iterate through all tokens.
    for i in range(n):
        # The token hasn't been encountered before.
        if cleanup.get(tokens[i]) == None:
            # Generate an atomic HC for the unencountered token.
            atomic_HC = thd.MAPTensor.random(1, dim)[0]
            # Add the atomic HC to the cleanup memory.
            cleanup[tokens[i]] = atomic_HC
            # Add the atomic (i.e., superpose) HC to the chunk HC representation.
            HC_representation += atomic_HC
        # The token has been encountered before.
        else:
            HC_representation += cleanup[tokens[i]]

    return HC_representation


def get_similarities_to_atomic_HVs(
    dim: int,
    cleanup: dict,
    memory: DSDM.DSDM,
    sentence: str,
    retrieve_mode: str = "pooling",
    k: int = None
) -> pd.DataFrame:
    
    # Actual inteference 
    retrieved_content = memory.retrieve(
        query_address=generate_query(
            dim,
            cleanup,
            preprocess.preprocess_text(sentence)[0]
        ),
        retrieve_mode=retrieve_mode,
        k=k,
    )

    if retrieve_mode == "pooling":
        sims_df = pd.DataFrame(
            columns=[
                'sentence',
                'token',
                'similarity'
            ]
        )
        
        for token, atomic_HC in cleanup.items():
            sims_df = pd.concat([
                sims_df,
                pd.DataFrame([
                    {
                        'sentence': sentence,
                        'token': token,
                        'similarity': thd.cosine_similarity(atomic_HC, retrieved_content).item()
                    }
                ])
            ])
        return sims_df
    else:
        return retrieved_content


def get_most_similar_HVs(
    sims_df: pd.DataFrame,
    delta_threshold: float = 0.15
) -> str:
    """
    """
    # Sort values: This is needed since similarity_next makes sense only in the context of a sort df.
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


def infer(
    dim: int,
    cleanup: dict,
    memory: DSDM.DSDM,
    inference_sentences: typing.List[str],
    retrieve_mode: str = "pooling",
    k=None,
    output=False
) -> typing.Union[pd.DataFrame, typing.List[typing.List]]:
    if retrieve_mode == "pooling":
        sims_df = pd.DataFrame(
            columns=[
                'sentence',
                'token',
                'similarity'
            ]
        ) 
        
        for inference_sentence in inference_sentences:
            sentence_sims_df = get_similarities_to_atomic_HVs(
                dim,
                cleanup,
                memory,
                inference_sentence,
                retrieve_mode
            ).sort_values('similarity', ascending=False).head(10)
            
            # Extract concept.
            #extracted_concept = get_most_similar_HVs(sentence_sims_df)
            
            sims_df = pd.concat([sims_df, sentence_sims_df])
            
        sims_df = (sims_df.sort_values(['sentence', 'similarity'], ascending=False) 
                          .set_index(['sentence', 'token'])
                  )
    
        if output:
            display(sims_df)
        return sims_df
    else:
        addresses = []

        for inference_sentence in inference_sentences:
            sentence_addresses = get_similarities_to_atomic_HVs(
                dim,
                cleanup,
                memory,
                inference_sentence,
                retrieve_mode,
                k
            )
            addresses.append(sentence_addresses)

        return addresses


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