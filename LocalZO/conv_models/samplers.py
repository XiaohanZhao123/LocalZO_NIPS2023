import torch
from abc import ABC, abstractmethod


class BaseSampler(ABC):

    def __init__(self):
        """abstract class for generating random tangents"""
        pass

    @abstractmethod
    def generate_random_tangents(self, inputs_shape: torch.Tensor, batch_size: int,
                                 sample_number: int, device) -> torch.Tensor:
        """
        generate random tangents accordingly, the output should be in shape [time_steps, sample_number, batch_size, *input_shape]
        """
        pass


class NormalSampler(BaseSampler):

    def __init__(self):
        """ sample the random tangents based on normal distribution"""
        super(NormalSampler, self).__init__()

    def generate_random_tangents(self, inputs_shape: torch.Tensor, batch_size, sample_number: int = 5, device='cpu') \
            -> torch.Tensor:
        assert inputs_shape[0] % batch_size == 0
        output_shape = (inputs_shape[0] // batch_size,) + (sample_number,) + (batch_size,) + inputs_shape[1:]
        return torch.randn(output_shape, device=device)
