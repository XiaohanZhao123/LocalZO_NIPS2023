import math
from typing import Any

import torch
from torch.autograd import Function


class PlainLIFFunction(Function):
    """plain implementation of LIF neuron, ignoring the sparsity, using atan for surrogate gradient"""

    @staticmethod
    def forward(ctx: Any, inputs, batch_size, u_th, beta) -> Any:
        ctx.constant = (batch_size, u_th, beta)
        inputs = inputs.view(-1, batch_size, *inputs.size()[1:])  # make it time first
        ctx.save_for_backward(torch.flip(inputs, dims=[0])) # since we need to compute the gradient in reverse order, we need to flip the inputs
        prev_memberance_potential = 0
        prev_output = 0
        outputs = []
        for input in inputs:
            current_memberance_potential = prev_memberance_potential * beta + input - prev_output * u_th
            output = (current_memberance_potential > u_th).float()
            outputs.append(output)
            prev_output = output
            prev_memberance_potential = current_memberance_potential

        outputs = torch.cat(outputs, dim=0)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # print('the sparsity of the tensor is', torch.sum(grad_outputs == 0) / torch.numel(grad_outputs))
        batch_size, u_th, beta = ctx.constant
        grad_outputs = grad_outputs.view(-1, batch_size, *grad_outputs.size()[1:])  # make it time first
        grad_outputs = torch.flip(grad_outputs, dims=[0])
        inputs = ctx.saved_tensors[0]
        # compute element-wise gradient
        grad_prev_memberance_potential = 0  # the gradient of the previous memberance potential
        grad_heavisides = atan_grad(inputs - u_th)
        grad_inputs = []
        for grad_output, grad_heaviside in zip(grad_outputs, grad_heavisides):
            grad_output = grad_output - grad_prev_memberance_potential * u_th
            grad_current_memberance_potential = grad_prev_memberance_potential * beta + grad_output * grad_heaviside
            grad_prev_memberance_potential = grad_current_memberance_potential
            # just append since partial[grad_membrane] / paritial[input] = 1
            grad_inputs.append(grad_current_memberance_potential)

        # flip the grad and return
        return torch.cat(grad_inputs[::-1], dim=0), None, None, None


class PlainLIFLocalZO(Function):
    """plain implementation of LIF neuron with ZO approximation for Heaviside function, ignoring the sparsity"""

    @staticmethod
    def forward(ctx: Any, inputs, batch_size, u_th, beta, random_tangents, delta) -> Any:
        """
        Forward pass of the LIF neuron with ZO approximation for Heaviside function

        Args:
            ctx: automatic differentiation context
            inputs: the input tensor with shape [batch_size * tim_step, *input_size]
            batch_size: batch size of each input, used to reshape the input
            u_th: the threshold of membrane potential for firing
            beta: drop rate for membrane potential
            random_tangents: random tangents to approximate the Heaviside function, with shape [time_step, sample_num, batch_size, *input_size]
            delta: to control the approximation error, should be a small positive number, smaller delta gives better approximation but larger machine error

        Returns: the output tensor with shape [batch_size * tim_step, *input_size]

        """
        ctx.constant = (batch_size, u_th, beta)
        inputs = inputs.view(-1, batch_size, *inputs.size()[1:])  # make it time first
        grad_heavisides = []  # gradients for the heaviside function
        prev_memberance_potential = 0
        prev_output = 0
        outputs = []
        for input, random_tangent in zip(inputs, random_tangents):
            current_memberance_potential = prev_memberance_potential * beta + input - prev_output * u_th
            output = (current_memberance_potential > u_th).float()
            outputs.append(output)

            # the shape of random_tangent is [sample_num, batch_size, *input_size]
            # providing mask for condition |u| < |z|delta in the paper
            g_2_mask = (torch.abs(current_memberance_potential - u_th) < torch.abs(random_tangent) * delta).float()
            g_2 = torch.abs(random_tangent) / 2 / delta * g_2_mask
            grad_heaviside = torch.mean(g_2, dim=0)
            grad_heavisides.append(grad_heaviside)
            prev_output = output
            prev_memberance_potential = current_memberance_potential

        outputs = torch.cat(outputs, dim=0)
        # since in bp we compute the last time step first, we need to flip gradient
        ctx.save_for_backward(torch.stack(grad_heavisides[::-1], dim=0))
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        """
        basically the same with PlainLIFFunction.backward, except that the gradient for heaviside is different

        Args:
            ctx: automatic differentiation context
            grad_outputs: the gradient of the output tensor with shape [batch_size * tim_step, *input_size]

        Returns:

        """
        batch_size, u_th, beta = ctx.constant
        grad_heavisides = ctx.saved_tensors[0] # gradients for heaviside function
        grad_outputs = grad_outputs.view(-1, batch_size, *grad_outputs.size()[1:])  # make it time first
        grad_outputs = torch.flip(grad_outputs, dims=[0])
        grad_prev_memberance_potential = 0  # the gradient of the previous memberance potential
        grad_inputs = []
        for grad_output, grad_heaviside in zip(grad_outputs, grad_heavisides):
            grad_output = grad_output - grad_prev_memberance_potential * u_th
            grad_current_memberance_potential = grad_prev_memberance_potential * beta + grad_output * grad_heaviside
            grad_prev_memberance_potential = grad_current_memberance_potential
            grad_inputs.append(grad_current_memberance_potential)

            # flip the grad and return
        return torch.cat(grad_inputs[::-1], dim=0), None, None, None, None, None



def fast_sigmoid(x):
    return x / (1 + torch.abs(x))


def atan_grad(x):
    alpha = 2.0
    grad = alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2))
    return grad


def sigmoid_grad(x):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)
