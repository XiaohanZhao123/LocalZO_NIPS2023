import torch


class ConstantEncodingCollate:
    def __init__(self, num_steps):
        """
        collate function for dataloader that encodes the input as a constant tensor of shape (num_steps, *input_shape)

        Args:
            num_steps:
        """
        self.num_steps = num_steps

    def __call__(self, batch):
        with torch.no_grad():
            if isinstance(batch[0], torch.Tensor):
                batch = torch.stack(batch)
                return batch
            elif isinstance(batch[0], tuple):
                transposed = zip(*batch)
                batch = [self(samples) for samples in transposed]
                return self.repeat(batch[0], self.num_steps), batch[1]
            else:
                return torch.tensor(batch)

    @staticmethod
    def repeat(tensor: torch.Tensor, num_steps):
        tensor = tensor.unsqueeze(0)
        return tensor.repeat(num_steps, *([1] * len(tensor.shape[1:])))


def repeat(tensor, num_steps):
    return tensor.unsqueeze(0).repeat(num_steps, *([1] * len(tensor.shape)))
