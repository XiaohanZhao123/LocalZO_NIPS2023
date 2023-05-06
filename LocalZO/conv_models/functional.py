import math
from typing import Any

import torch
from torch.autograd import Function

lif_save_dirctory = './lif.txt'
lif_zo_save_dirctory = './lifzo.txt'

layer_idx = 0

class PlainLIFFunction(Function):
    """plain implementation of LIF neuron, ignoring the sparsity, using atan for surrogate gradient"""

    @staticmethod
    def forward(ctx: Any, inputs, batch_size, u_th, beta) -> Any:
        ctx.constant = (batch_size, u_th, beta)
        inputs = inputs.view(-1, batch_size, *inputs.size()[1:])  # make it time first
        prev_memberance_potential = 0
        prev_output = 0
        outputs = []
        mem_rec = []  # the membrane potential record, used to compute the gradient
        for input in inputs:
            current_memberance_potential = prev_memberance_potential * beta + input - prev_output * u_th
            output = (current_memberance_potential > u_th).float()
            outputs.append(output)
            prev_output = output
            prev_memberance_potential = current_memberance_potential
            mem_rec.append(current_memberance_potential)

        ctx.save_for_backward(torch.stack(mem_rec[::-1], dim=0) - u_th)
        outputs = torch.cat(outputs, dim=0)
        outputs.view(-1)[0] += 1
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        global layer_idx
        layer_idx += 1
        # print('the sparsity of the tensor is', torch.sum(grad_outputs == 0) / torch.numel(grad_outputs))
        batch_size, u_th, beta = ctx.constant
        grad_outputs = grad_outputs.view(-1, batch_size, *grad_outputs.size()[1:])  # make it time first
        grad_outputs = torch.flip(grad_outputs, dims=[0])
        print(f'layer_idx: {layer_idx}, sparsity_of_input_grad: {compute_sparsity(grad_outputs)}')
        mem_rec = ctx.saved_tensors[0]
        # compute element-wise gradient
        grad_prev_memberance_potential = 0  # the gradient of the previous memberance potential
        grad_heavisides = atan_grad(mem_rec)
        grad_inputs = []
        for grad_output, grad_heaviside in zip(grad_outputs, grad_heavisides):
            grad_output = grad_output - grad_prev_memberance_potential * u_th
            grad_current_memberance_potential = grad_prev_memberance_potential * beta + grad_output * grad_heaviside
            grad_prev_memberance_potential = grad_current_memberance_potential
            # just append since partial[grad_membrane] / paritial[input] = 1
            grad_inputs.append(grad_current_memberance_potential)

        # flip the grad and return
        grad_inputs = torch.cat(grad_inputs[::-1], dim=0)
        print(f'layer_idx: {layer_idx}, sparsity_of_output_grad: {compute_sparsity(grad_inputs)}')
        return grad_inputs, None, None, None


class PlainLIFLocalZOMultiSample(Function):
    """plain implementation of LIF neuron with ZO approximation for Heaviside function, ignoring the sparsity, using multi sample to approximate gradient"""

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
        mem_rec = []  # record the membrane potential, used to compute the gradient
        for input in inputs:
            current_memberance_potential = prev_memberance_potential * beta + input - prev_output * u_th
            output = (current_memberance_potential > u_th).float()
            outputs.append(output)
            mem_rec.append(current_memberance_potential)
            prev_output = output
            prev_memberance_potential = current_memberance_potential

        # providing mask for condition |u| < |z| * delta in the paper, also, we flip the gradient
        u = torch.stack(mem_rec[::-1], dim=0) - u_th
        g_2_mask = (torch.abs(u) < torch.abs(random_tangents) * delta).float()
        g_2 = torch.abs(random_tangents) / 2 / delta * g_2_mask
        ctx.save_for_backward(torch.mean(g_2, dim=0))
        outputs = torch.cat(outputs, dim=0)
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
        grad_heavisides = ctx.saved_tensors[0]  # gradients for heaviside function
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


class PlainLIFLocalZOOnce(Function):
    """plain implementation of LIF neuron with ZO approximation for Heaviside function, ignoring the sparsity, using multi sample to approximate gradient"""

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
        prev_memberance_potential = 0
        prev_output = 0
        outputs = []
        mem_rec = []  # record the membrane potential, used to compute the gradient
        for input in inputs:
            current_memberance_potential = prev_memberance_potential * beta + input - prev_output * u_th
            output = (current_memberance_potential > u_th).float()
            mem_rec.append(current_memberance_potential)
            outputs.append(output)
            prev_output = output
            prev_memberance_potential = current_memberance_potential

        # providing mask for condition |u| < |z| * delta in the paper, also, we flip the gradient
        u = torch.abs(torch.stack(mem_rec[::-1], dim=0) - u_th)
        random_tangents.abs_()
        g_2_mask = (u < random_tangents * delta).float()
        g_2 = random_tangents.div_(2 * delta).mul_(g_2_mask)
        ctx.save_for_backward(g_2)
        outputs = torch.cat(outputs, dim=0)
        outputs.view(-1)[0] += 1
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
        global layer_idx
        batch_size, u_th, beta = ctx.constant
        grad_heavisides = ctx.saved_tensors[0]  # gradients for heaviside function
        grad_outputs = grad_outputs.view(-1, batch_size, *grad_outputs.size()[1:])  # make it time first
        print(f'layer_idx: {layer_idx}, sparsity_of_input_grad: {compute_sparsity(grad_outputs)}')
        grad_outputs = torch.flip(grad_outputs, dims=[0])
        grad_prev_memberance_potential = 0  # the gradient of the previous memberance potential
        grad_inputs = []
        for grad_output, grad_heaviside in zip(grad_outputs, grad_heavisides):
            grad_output = grad_output - grad_prev_memberance_potential * u_th
            grad_current_memberance_potential = grad_prev_memberance_potential * beta + grad_output * grad_heaviside
            grad_prev_memberance_potential = grad_current_memberance_potential
            grad_inputs.append(grad_current_memberance_potential)

            # flip the grad and return
        grad_inputs = torch.cat(grad_inputs[::-1], dim=0)
        print(f'layer_idx: {layer_idx}, sparsity_of_output_grad: {compute_sparsity(grad_inputs)}')
        return grad_inputs, None, None, None, None, None



class PlainLIFFunctionProfile(Function):
    """plain implementation of LIF neuron, ignoring the sparsity, using atan for surrogate gradient"""

    @staticmethod
    def forward(ctx: Any, inputs, batch_size, u_th, beta) -> Any:
        ctx.constant = (batch_size, u_th, beta)
        inputs = inputs.view(-1, batch_size, *inputs.size()[1:])  # make it time first
        prev_memberance_potential = 0
        prev_output = 0
        outputs = []
        mem_rec = []  # the membrane potential record, used to compute the gradient
        for input in inputs:
            current_memberance_potential = prev_memberance_potential * beta + input - prev_output * u_th
            output = (current_memberance_potential > u_th).float()
            outputs.append(output)
            prev_output = output
            prev_memberance_potential = current_memberance_potential
            mem_rec.append(current_memberance_potential)

        ctx.save_for_backward(torch.stack(mem_rec[::-1], dim=0) - u_th)
        outputs = torch.cat(outputs, dim=0)
        return outputs

    @staticmethod
    def backward(ctx: Any, grad_outputs: Any) -> Any:
        # print('the sparsity of the tensor is', torch.sum(grad_outputs == 0) / torch.numel(grad_outputs))
        batch_size, u_th, beta = ctx.constant
        grad_outputs = grad_outputs.view(-1, batch_size, *grad_outputs.size()[1:])  # make it time first
        print('the sparsity of the input grad is', compute_sparsity(grad_outputs))
        grad_outputs = torch.flip(grad_outputs, dims=[0])
        mem_rec = ctx.saved_tensors[0]
        # compute element-wise gradient
        grad_prev_memberance_potential = 0  # the gradient of the previous memberance potential
        grad_heavisides = atan_grad(mem_rec)
        grad_inputs = []
        for grad_output, grad_heaviside in zip(grad_outputs, grad_heavisides):
            grad_output = grad_output - grad_prev_memberance_potential * u_th
            grad_current_memberance_potential = grad_prev_memberance_potential * beta + grad_output * grad_heaviside
            grad_prev_memberance_potential = grad_current_memberance_potential
            # just append since partial[grad_membrane] / paritial[input] = 1
            grad_inputs.append(grad_current_memberance_potential)

        # flip the grad and return
        grad_inputs = torch.cat(grad_inputs[::-1], dim=0)
        print('the sparsity of the output grad is', compute_sparsity(grad_inputs))
        return grad_inputs, None, None, None


class PlainLIFLocalZOOnceProfile(Function):
    """plain implementation of LIF neuron with ZO approximation for Heaviside function, ignoring the sparsity, using multi sample to approximate gradient"""

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
        mem_rec = []  # record the membrane potential, used to compute the gradient
        for input in inputs:
            current_memberance_potential = prev_memberance_potential * beta + input - prev_output * u_th
            output = (current_memberance_potential > u_th).float()
            mem_rec.append(current_memberance_potential)
            outputs.append(output)
            prev_output = output
            prev_memberance_potential = current_memberance_potential

        # providing mask for condition |u| < |z| * delta in the paper, also, we flip the gradient
        u = torch.abs(torch.stack(mem_rec[::-1], dim=0) - u_th)
        random_tangents.abs_()
        g_2_mask = (u < random_tangents * delta).float()
        g_2 = random_tangents / 2 / delta * g_2_mask
        ctx.save_for_backward(g_2)
        outputs = torch.cat(outputs, dim=0)
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
        grad_heavisides = ctx.saved_tensors[0]  # gradients for heaviside function
        grad_outputs = grad_outputs.view(-1, batch_size, *grad_outputs.size()[1:])  # make it time first
        print('the sparsity of the input grad is', compute_sparsity(grad_outputs))
        grad_outputs = torch.flip(grad_outputs, dims=[0])
        grad_prev_memberance_potential = 0  # the gradient of the previous memberance potential
        grad_inputs = []
        for grad_output, grad_heaviside in zip(grad_outputs, grad_heavisides):
            grad_output = grad_output - grad_prev_memberance_potential * u_th
            grad_current_memberance_potential = grad_prev_memberance_potential * beta + grad_output * grad_heaviside
            grad_prev_memberance_potential = grad_current_memberance_potential
            grad_inputs.append(grad_current_memberance_potential)

            # flip the grad and return
        grad_inputs = torch.cat(grad_inputs[::-1], dim=0)
        print('the sparsity of the output grad is', compute_sparsity(grad_inputs))
        return grad_inputs, None, None, None, None, None



def fast_sigmoid(x):
    return x / (1 + torch.abs(x))


def atan_grad(x):
    alpha = 2.0
    grad = alpha / 2 / (1 + (math.pi / 2 * alpha * x).pow_(2))
    return grad


def sigmoid_grad(x):
    sigmoid_x = torch.sigmoid(x)
    return sigmoid_x * (1 - sigmoid_x)


def compute_sparsity(x):
    return torch.sum((x == 0).float()) / x.numel()
