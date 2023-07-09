"""Models for DSDM."""
from abc import abstractmethod
from better_abc import ABCMeta, abstract_attribute
import torch
import torchhd as thd
from typing import List


class Query():
    HVs: List[torch.Tensor]
    query_address: torch.Tensor


class Model(metaclass=ABCMeta):
    @abstract_attribute
    def dim_address(self):
        pass

    @abstract_attribute
    def dim_content(self):
        pass

    @abstractmethod
    def update_memory_address(self, query_address, memory_address, weight):
        pass

    @abstractmethod
    def update_memory_content(self, query_content, memory_content, weight):
        pass


class PositionalModel(Model):
    def __init__(self, dim_address, chunk_length):
        self.dim_address = dim_address
        self.dim_content = None

        self.chunk_length = chunk_length
        # Generate positional HVs.
        self.positional_HVs = thd.BSCTensor.random(dim_address, chunk_length - 1)
        # Generate HV for tie breaking in superposition.
        self.tie_breaker_HV = thd.BSCTensor.random(dim_address, 1)

    def normalize(self, representation: torch.Tensor, numbers_HVs: int) -> torch.Tensor:
        """Implement majority rule as normalization."""
        threshold = numbers_HVs // 2
        return torch.greater(representation, threshold)

    def update_memory_address(self, query: Query, memory_address_unnormalized: torch.Tensor, weight: int):
       # Normalize memory address.
       memory_address = self.normalize(memory_address_unnormalized)
       for positional_HV in self.positional_HVs:
           ubinding = memory_address.bindpositional_HV
           sim = 1

    def update_memory_content(self, query_address, memory_address, weight):
        """This model is used only in the context of autoassociative memory."""
        return None

    def create_query_representation(self, HVs: List[thd.VSATensor]):
        query_representation = thd.identity(1, self.dim_address, "BSC")
        number_HVs = len(HVs)

        for HV, positional_HV in zip(HVs[:-1], self.positional_HVs):
            # Superpose the bindings of the positional HVs with the token HVs.
            query_representation = torch.add(query_representation, positional_HV.bind(HV).long())
        # Break ties in case of superposition of an even no. of HVs.
        if len(HVs) % 2 == 0:
            query_representation = torch.add(query_representation, self.tie_breaker_HV.long())
            number_HVs += 1
        # Normalize superposition by integer-boolean conversion.
        query_representation = self.normalize(query_representation, number_HVs)
        
        return query_representation



def main():
    x = thd.identity(1, 10, "MAP")

    print(x)




if __name__ == "__main__":
    main()