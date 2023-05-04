gpu_idx = '3'
import os

os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx
import torch
import matplotlib
from experiments.models.vgg import VGGSpconv, VGGSNNTorch
from dataset.cifar10 import get_dataset
from utils import forward_snntorch, train_and_profile_snntorch, train_and_profile_spconv
from snntorch import functional as SF

data_root = '/home/zxh/remote/async_fgd/experiment_resources/dataset/cifar10'
batch_size = 4
num_steps = 6
u_th = 0.3
beta = 0.6
in_channel = 3
num_class = 10

if __name__ == '__main__':
    train_loader = get_dataset(root=data_root, batch_size=batch_size, encoding='constant', num_steps=num_steps)
    spconv_model = VGGSpconv(u_th=u_th, beta=beta, batch_size=batch_size, num_steps=num_steps,
                             in_channel=in_channel, num_class=num_class).cuda()
    # snntorch_model = VGGSNNTorch(u_th=u_th, beta=beta, in_channel=in_channel, num_class=num_class).cuda()
    # train_and_profile_snntorch(snntorch_model, train_loader, SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2),
    #                            SF.accuracy_rate, )
    # del snntorch_model
    torch.cuda.empty_cache()
    train_and_profile_spconv(spconv_model, train_loader, SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2),
                             SF.accuracy_rate, )
