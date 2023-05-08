import os
import time

import snntorch.functional as SF
import torch
from snntorch import utils
from torch import nn

from LocalZO.conv_models.neurons import LeakyPlain, LeakyZOPlainOnce
from dataset.utils import repeat


def forward_snntorch(net, data):
    spk_rec = []
    utils.reset(net)

    for step in range(data.size(0)):
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec, dim=0)


def real_reset(net: nn.Module):
    """
    traver the network recursively and set the spiking layer's spike and membrane potential to zero

    Args:
        net:

    Returns:

    """

    for layer in net.children():
        if isinstance(layer, torch.nn.Sequential):
            if isinstance(layer[0], torch.nn.Sequential):
                real_reset(layer)  # the utils.reset can only reset module with depth=1, so we to go deeper
            else:
                utils.reset(layer)
        else:
            utils.reset(layer)


def train_and_profile_snntorch(net,
                               train_loader,
                               loss_fn=SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2),
                               acc_fn=SF.accuracy_rate,
                               num_epochs=2,
                               lr=2e-3,
                               constant_encoding=False,
                               num_steps=6):
    if constant_encoding is True:
        assert num_steps > 0, 'num_steps must be greater than 0 when constant_encoding is True'
    max_iter = 10
    time_list = []
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for i, (x, labels) in enumerate(train_loader):
            if i > max_iter:
                break
            if constant_encoding:
                x = repeat(x, num_steps)
            net.train()
            optimizer.zero_grad()
            x = x.cuda()
            labels = labels.cuda()
            torch.cuda.synchronize()
            start = time.time()
            spk_out = forward_snntorch(net, x)
            loss = loss_fn(spk_out, labels)
            loss.backward()
            # optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            if i > 0:
                time_list.append(end - start)
            # print(f'epoch: {epoch}, iter: {i}, loss: {loss.item()}, acc: {acc_fn(spk_out, labels)}')

    mean_time = sum(time_list) / len(time_list)
    print(f'snn torch mean_time: {mean_time}')
    return mean_time


def train_and_profile_spconv(net,
                             train_loader,
                             loss_fn=SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2),
                             acc_fn=SF.accuracy_rate,
                             num_epochs=2,
                             lr=2e-3,
                             constant_encoding=False,
                             num_steps=6,
                             save_dir='./spconv_model.pth'):
    if save_dir is not None and os.path.exists(save_dir):
        net.load_state_dict(torch.load(save_dir))
    if constant_encoding is True:
        assert num_steps > 0, 'num_steps must be greater than 0 when constant_encoding is True'
    max_iter = 10
    time_list = []
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(num_epochs):
        for i, (x, labels) in enumerate(train_loader):
            if i > max_iter:
                break
            if constant_encoding:
                x = repeat(x, num_steps)

            optimizer.zero_grad()
            x = x.cuda()
            labels = labels.cuda()
            torch.cuda.synchronize()
            start = time.time()
            output = net(x)
            loss = loss_fn(output, labels)
            # acc = acc_fn(output, labels)
            loss.backward()
            # optimizer.step()
            torch.cuda.synchronize()
            end = time.time()
            if i > 0:
                time_list.append(end - start)
            # print(f'epoch: {epoch}, iter: {i}, loss: {loss.item()}, acc: {acc_fn(output, labels)}')

    if save_dir:
        torch.save(net.state_dict(), save_dir)
    mean_time = sum(time_list) / len(time_list)
    print(f'spconv mean_time: {mean_time}')
    return mean_time


def replace_leaky_zo_plain_once(model: torch.nn.Module):
    """
    Replace LeakyZOPlainOnce with LeakyPlain in the given model.

    Args:
        model: The model to be modified.

    Returns: None, in-place changes.
    """
    for name, module in model.named_children():
        if isinstance(module, LeakyZOPlainOnce):
            # Replace LeakyZOPlainOnce with LeakyPlain
            u_th = module.u_th
            beta = module.beta
            batch_size = module.batch_size
            plain_module = LeakyPlain(u_th, beta, batch_size)
            setattr(model, name, plain_module)
        else:
            # Recursively replace LeakyZOPlainOnce with LeakyPlain in the child module
            replace_leaky_zo_plain_once(module)

