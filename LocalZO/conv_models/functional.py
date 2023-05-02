from typing import Any

import torch
from torch.autograd import Function


class PlainLIFFunction(Function):
    """plain implementation of LIF neuron, ignoring the sparsity"""

    @staticmethod
    def forward(ctx: Any, inputs, batch_size, u_th, beta) -> Any:
        print('test forward')
        ctx.constant = (batch_size, u_th, beta)
        inputs = inputs.view(-1, batch_size, *inputs.size()[1:])  # make it time first
        ctx.save_for_backward(torch.flip(inputs, dims=[0]))
        prev_memberance_potential = 0
        prev_output = 0
        outputs = []
        for input in inputs:
            current_memberance_potential = prev_memberance_potential * beta + input - prev_output * u_th
            output = (current_memberance_potential > u_th).float()
            outputs.append(output)
            prev_output = output

        outputs = torch.concat(outputs, dim=0)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        print('test backward!')
        print('got', grad_outputs, 'as input')
        print('the sparsity of the tensor is', torch.sum(grad_outputs == 0) / torch.numel(grad_outputs))
        batch_size, u_th, beta = ctx.constant
        grad_outputs.view(-1, batch_size, *grad_outputs.size()[1:])  # make it time first
        grad_outputs = torch.flip(grad_outputs, dims=[0])
        inputs = ctx.saved_tensors[0]
        # compute element-wise gradient
        grad_prev_memberance_potential = 0 # the gradient of the previous memberance potential
        sigmoid = torch.sigmoid(inputs - u_th)
        grad_heavisides = sigmoid * (1 - sigmoid)
        grad_inputs = []
        for grad_output, grad_heaviside in zip(grad_outputs, grad_heavisides):
            grad_output = grad_output - grad_prev_memberance_potential * u_th
            grad_current_memberance_potential = grad_prev_memberance_potential * beta + grad_output * grad_heaviside
            grad_prev_memberance_potential = grad_current_memberance_potential
            grad_inputs.append(grad_current_memberance_potential)

        return torch.concat(grad_inputs, dim=0), None, None, None
