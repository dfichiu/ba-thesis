"""This file implements DSDM."""
import torch
import torch.nn as nn
import torch.nn.functional as F 


# Torch settings: Disable gradients.
torch.set_grad_enabled(False)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DSDM(nn.Module):
    """
    Dynamic Sparse Distributed Memory (DSDM) as introduced in
    https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850721.pdf.
    
    DSDM is a dynamic version of Pentti Kanerva's SDM, which is biologically
    plausible computer memory model.
    
    The implementation here differs in three main respects from the algorithm
    in https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136850721.pdf:
    i) It uses the correct definition of the exponential moving average (
    recursive temperature), which is used as a dynamic threshold. The same defintion is
    also used by the authors in their experiment code.
    ii) In the memory update step, it substract the memory address from the query 
    address. The same substraction is also used by the authors in their experiment code.
    iii) The pruning method implments the frequecy-based pruning mechanism introduced
    in the thesis.
    
    Furthermore, there is no content, but each address is associated with to scores:
    i) chunk score: If the addresses are built from Transformer self-attention matrices,
      the avg. of the associated attention weights. Otherwise, 0.
    ii) bin score: Running sum of address "softmin weights."
    
    Attributes:
        address_size:
            Size of the hypervectors.
        ema_time_period:
            Number of days of the exponential moving average (EMA).
        learning_rate_update:
            Learning rate used in the update step.
        temperature:
            Base of the softmin exponent.
        normalize:
            If True, normalize addresses in cosine similarity computation.
        prune_mode:
            If 'fixed_size' or 'remove_percentage,' prune memory.
        pruning_frequency_type:
            'Document' or 'sentence'
        pruning_frequency:
            Number of documents/sentences after which to prune the memory.
        max_size_address_space:
            If prune mode is 'fixed_size,' the maximum size of the address space.
        remove_percentage:
            Percentage of addresses to remove.
        safeguard_bins:
            If True, do not remove addresses with a bin score lower than bin_score_threshold.
        bin_score_threshold:
            If safegurd_bins is True and bin_score_threshold_type is 'static,' bin score threshold.
        bin_score_threshold_type:
            If safegurd_bins is True, type of bin threshold: 'static' or 'dynamic.'
        safeguard_chunks:
            If True, do not removed addresses with a chunk score lower then chunk_score_threshold.
        chunk_score_threshold:
            If safeguard_chunks is True, chunk score threshold.
        chunk_size: If the sliding window n-gram method is used, the n-gram dimension used.
    """
    
    def __init__(
        self,
        address_size,
        ema_time_period,
        learning_rate_update,
        temperature,
        normalize=False,
        prune_mode=None,
        pruning_frequency_type=None,
        pruning_frequency=None,
        max_size_address_space=None,
        remove_percentage=None,
        safeguard_bins=False,
        bin_score_threshold=0,
        bin_score_threshold_type='static',
        safeguard_chunks=False,
        chunk_score_threshold=0,
        chunk_size=None,
    ):
        super(DSDM, self).__init__()
        self.address_size = address_size
        self.addresses = torch.tensor([]).to(device)
        self.scores = torch.tensor([]).to(device)  

        self.normalize = normalize

        self.ema = 0
        self.ema_time_period = ema_time_period
        self.ema_temperature = 2 / (self.ema_time_period + 1)
        
        self.learning_rate_update = learning_rate_update

        self.temperature = temperature
        
        
        # Deleted addresses 
        self.deleted_addresses = torch.tensor([]).to(device)
        self.deleted_scores = torch.tensor([]).to(device)  
        

        # Set statistics counters.
        self.n_updates = 0
        self.n_expansions = 0
        self.n_deletions = 0
        
        # Sliding window n-gram method variables
        self.removed_duplicates = 0
        self.chunk_size = chunk_size

        # Set pruning hyperparameters.
        self.prune_mode = prune_mode
        self.max_size_address_space = max_size_address_space
        self.remove_percentage = remove_percentage
        
        self.pruning_frequency_type = pruning_frequency_type
        self.pruning_frequency = pruning_frequency
        
        self.safeguard_bins = safeguard_bins
        self.bin_score_threshold_type = bin_score_threshold_type
        self.bin_score_threshold = bin_score_threshold
        
        self.safeguard_chunks = safeguard_chunks
        self.chunk_score_threshold = chunk_score_threshold

        # Set wiki article index.
        self.wiki_articles = torch.tensor([]).to(device)

        
    def add_wiki_article(self, article_id: int) -> None:
        self.wiki_articles = torch.cat(
            (
                self.wiki_articles,
                torch.tensor([article_id]).to(device)
            )
        )
        return
        
        
    def get_memory_type(self) -> str:
        return "normalized" if self.normalize == True else "unnormalized"

    
    def set_temperature(self, temperature):
        self.temperature = temperature

    
    def set_learning_rate_update(self, learing_rate_update):
        self.learning_rate_update = learning_rate_update
        
        
    def retrieve(
        self,
        query_address: torch.Tensor,
        retrieve_mode: str = 'pooling',
        k = None
    ):
        """
        Queries memory and retrieves content.
        
        Args:
            query_address: Token superposition.
            retrieve_mode: If 'pooling', the entire memory space is summed up
              as per the original DSDM retrieve operation. Else, the
              closest k addresses to the query_address are returned.
            k: The number of returned addresses when retrieve_mode is not 'pooling.'
        """
        query_address = query_address.to(device)
#         Prune before retrieval.
#         self.prune()

        cos = torch.nn.CosineSimilarity()
        # Calculate the cosine similarities.
        if self.normalize: 
            similarities = cos(self.addresses.sgn(), query_address.sgn())
        else:
            similarities = cos(self.addresses, query_address)
        # Cosine distance tensor
        distances = 1 - similarities

        # Calculate the softmin weights.
        softmin_weights = F.softmin(distances / self.temperature, dim=-1)

        if retrieve_mode == "pooling":
            # Weight the memory addresses with the softmin weights.
            weighted_addresses = torch.matmul(softmin_weights, self.addresses.to(device)).view(-1)

            # Pool the weighted memory addresses to create the output and return it.
            return torch.sum(weighted_addresses.view(1, -1), 0)
        else:  # retrieve_mode == "top_k"
            
            # Make sure the number of to return addresses is not
            # bigger than the total number of memory addresses.
            k = k if k <= len(self.addresses) else len(self.addresses)
            
            # Get k closest addresses.
            val, idx = torch.topk(
                similarities.view(1, -1),
                k=k,
                largest=True
            )

            return self.addresses[idx[0]]
   
    
    def save(self, query_address: torch.Tensor, chunk_score: float = 0):
        """
        Saves query address to memory.
        
        If the query is close enogh to one of the memory addresses,
        the memory space is updated. Otherwise, the memory space
        is extended.
        
        Args:
            query_address: Token superposition.
            chunk_score: If the token superposition comes from a self-attention
              matrix, the avg. of the associated attention weights.
        """
        query_address = query_address.to(device)
        
        if self.addresses.shape[0] == 0:
            # The memory is instantiated with the first observation.
            self.addresses = torch.cat(
                (
                    self.addresses,
                    query_address.view(1, -1)
                )
            )
            self.scores = torch.cat(
                (
                    self.scores,
                    torch.tensor([chunk_score, 0]).view(1, -1).to(device)
                )
            )
            self.n_expansions += 1  
            return
        
        cos = torch.nn.CosineSimilarity()
        # Calculate the cosine similarities.
        if self.normalize: 
            similarities = cos(self.addresses.sgn(), query_address.sgn())
        else:
            similarities = cos(self.addresses, query_address)

        # Calculate the cosine distances.
        distances = 1 - similarities
        # Get the minimum distance and the corresponding address index.  
        min_distance = torch.min(distances, dim=0)[0].item()
        
        # Calculate EMA for current chunk.
        self.ema += self.ema_temperature * (min_distance - self.ema)
#         print(f'Min. distance: {min_distance}')
#         print(f'EMA: {self.ema}')
        
        # Check if the minimum distance is bigger than the adaptive threshold.
        if min_distance > self.ema: # If the minimum distance is bigger, create a new address.
            # Add a new entry to the address matrix/tensor equal to the target address.
            self.addresses = torch.cat(
                (
                    self.addresses,
                    query_address.view(1, -1)
                )
            )
            self.scores = torch.cat(
                (
                    self.scores,
                    torch.tensor([chunk_score, 0]).view(1, -1).to(device)
                )
            )
#             print(f"Address index: {self.n_expansions}")
            self.n_expansions += 1  
        else: # If the minimum distance is smaller or equal, update the memory addresses.
            # Apply the softmin function to the distance tensor the get the softmin weights.
            softmin_weights = F.softmin(distances / self.temperature, dim=-1)
            # Update the memory address space.
            self.addresses += self.learning_rate_update * torch.mul(softmin_weights.view(-1, 1), query_address - self.addresses)
#             print(self.addresses[len(self.addresses) - 1])
            self.scores[:, 1] += softmin_weights
            self.n_updates += 1
            
            if self.safeguard_bins and self.bin_score_threshold_type == "dynamic":
                val, idx = torch.topk(
                    self.scores[:, 1],
                    k=10,
                    largest=False,
                )
                self.bin_score_threshold += val[-1].item()
            
        return

    
    def prune(self):
        """
        Prunes memory space.
        """
        if self.prune_mode is not None: 
            if (
                (
                  self.prune_mode == "fixed-size"
                  and (len(self.addresses) > self.max_size_address_space)
                )
                or self.prune_mode == "remove-percentage"
            ):
                # Bin score ascending sorting
                inner_sorting = torch.argsort(self.scores[:, 1])
                # Chunk score ascending sorting
                outer_sorting = torch.argsort(
                    self.scores[inner_sorting][:, 0], stable=True
                )
                
                # Sort scores and addresses.
                self.scores = self.scores[inner_sorting][outer_sorting]
                self.addresses = self.addresses[inner_sorting][outer_sorting]
                
                if self.prune_mode == "fixed-size":
                    n_keep = self.max_size_address_space
                else:
                    n_keep = int((1 - self.remove_percentage) * len(self.addresses))

                if self.safeguard_bins or self.safeguard_chunks:  # Bin or chunk safeguarding
                    if self.safeguard_bins and self.safeguard_chunks: 
                        keep_mask = (
                            (self.scores[:, 1] <= self.bin_score_threshold)
                            | (self.scores[:, 0] >= self.chunk_score_threshold)
                        )
                    elif self.safeguard_bins:
                        keep_mask = self.scores[:, 1] <= self.bin_score_threshold

                    else:
                        keep_mask = self.scores[:, 0] >= self.chunk_score_threshold
                        
                    keep_mask[-n_keep:] = True
                    
                    
                    # Update deleted addresses and scores.
                    self.deleted_addresses = torch.cat(
                        (
                            self.deleted_addresses,
                            self.addresses[~keep_mask],
                        )
                    )
                    self.deleted_scores = torch.cat(
                        (
                             self.deleted_scores,
                            self.scores[~keep_mask],
                        )
                    )
                    
                    
                    # Update number of deleted addresses.
                    self.n_deletions += torch.sum(~keep_mask).item()
                    
                    # Update memory.
                    self.scores = self.scores[keep_mask]
                    self.addresses = self.addresses[keep_mask]
                else:  # No bin or chunk safeguarding
                    # Update number of deleted addresses.
                    self.n_deletions += len(self.addresses) - n_keep 
                    
                    # Update deleted addresses and scores.
                    self.deleted_addresses = torch.cat(
                        (
                            self.deleted_addresses,
                            self.addresses[:-n_keep],
                        )
                    )
                    self.deleted_scores = torch.cat(
                        (
                             self.deleted_scores,
                            self.scores[:-n_keep],
                        )
                    )
                    
                    # Update memory.
                    self.scores = self.scores[-n_keep:]
                    self.addresses = self.addresses[-n_keep:]
                    
        return
