"""This file implements the functions needed to build subsequences from a weighted directed graph."""
import networkx as nx
import numpy as np
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def backward_pass(G, current_node, left_edge, right_edge, sequence, mean, sequences, means):
    in_nodes = np.array([edge[0] for edge in list(G.in_edges(current_node))])
    in_nodes = in_nodes[(in_nodes > left_edge) & (in_nodes < current_node)]
    for node in in_nodes:
        sequence[node] = 1
        sequences.append(sequence)
        mean += G[node][current_node]['weight']
        means.append(round(mean / (sum(sequence) - 1), 2))
        backward_pass(G, node, left_edge, node, sequence.copy(), mean, sequences, means)
        forward_pass(G, node, left_edge, current_node, sequence.copy(), mean, sequences, means)
        
    return
    
    
def forward_pass(G, current_node, left_edge, right_edge, sequence, mean, sequences, means):
    out_nodes = np.array([edge[1] for edge in list(G.out_edges(current_node))])
    out_nodes = out_nodes[(out_nodes > current_node) & (out_nodes < right_edge)]
    for node in out_nodes:
        sequence[node] = 1
        sequences.append(sequence)
        mean += G[current_node][node]['weight']
        means.append(round(mean / (sum(sequence) - 1), 2))
        backward_pass(G, node, current_node, node, sequence.copy(), mean, sequences, means)
        forward_pass(G, node, node, right_edge, sequence.copy(), mean, sequences, means)
            
    return
    

def construct_sequences(G: nx.DiGraph, n_tokens: int):
    sequences = []
    means = []
    
    for node in G.nodes():
        sequence = np.zeros(n_tokens)
        mean = 0
        sequence[node] = 1
        #sequences.append(sequence) # Do not allow for 1-token sequences.
        forward_pass(G, node, node, n_tokens, sequence.copy(), mean, sequences,  means)

    return means, sequences