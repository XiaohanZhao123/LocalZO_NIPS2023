from snntorch import utils
import torch
import snntorch
import time


def forward_snntorch(net, data):
    spk_rec = []
    utils.reset(net)

    for step in range(data.size(0)):
        spk_out, mem_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec, dim=0)


def train_and_profile_snntorch(net, train_loader, loss_fn, acc_fn, num_epochs=2, lr=2e-3):
    max_iter = 4
    time_list = []
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for i, (x, labels) in enumerate(train_loader):
            if i > max_iter:
                break
            optimizer.zero_grad()
            net.train()
            start = time.time()
            x = x.cuda()
            labels = labels.cuda()
            spk_out = forward_snntorch(net, x)
            loss = loss_fn(spk_out, labels)
            loss.backward(retain_graph=True)
            optimizer.step()
            end = time.time()
            if i > 0:
                time_list.append(end - start)
            print(f'epoch: {epoch}, iter: {i}, loss: {loss.item()}')

        mean_time = sum(time_list) / len(time_list)
        print(f'epoch: {epoch}, mean_time: {mean_time}')


def train_and_profile_spconv(net, train_loader, loss_fn, acc_fn, num_epochs=2, lr=2e-3):
    max_iter = 4
    time_list = []
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    for epoch in range(num_epochs):
        for i, (x, labels) in enumerate(train_loader):
            if i > max_iter:
                break
            start = time.time()
            optimizer.zero_grad()
            x = x.cuda()
            labels = labels.cuda()
            output = net(x)
            loss = loss_fn(output, labels)
            # acc = acc_fn(output, labels)
            loss.backward()
            optimizer.step()
            end = time.time()
            if i > 0:
                time_list.append(end - start)
            print(f'epoch: {epoch}, iter: {i}, loss: {loss.item()}')

        mean_time = sum(time_list) / len(time_list)
        print(f'epoch: {epoch}, mean_time: {mean_time}')
