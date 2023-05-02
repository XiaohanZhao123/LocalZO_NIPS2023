import torch


def generate_sparse_input(shape, sparsity):
    outputs = torch.rand(shape)
    num_of_zeros = int(outputs.numel() * sparsity)
    indices = torch.randperm(outputs.numel())[:num_of_zeros]
    outputs.view(-1)[indices] = 0
    return outputs
