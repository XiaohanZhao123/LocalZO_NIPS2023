import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import snntorch.functional
import torch
import hydra
from hydra.utils import call
from dataset.utils import repeat
from utils import replace_leaky_zo_plain_once
from LocalZO.conv_models.functional import layer_index_reset, set_output_dir

torch.manual_seed(0)

# dataloader arguments
batch_size = 128
data_path = '/home/zxh/data'
loss_fn = snntorch.functional.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
output_head = '/home/zxh/remote/sparse_zo_nips2023/experiments/results/grad_sparsity'


@hydra.main(config_path='config', config_name='defaults', version_base=None)
def main(cfg):
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    print(f'testing on {model_name}, {dataset_name}')
    print(f'model setup: u_th:{cfg.model.u_th}, beta:{cfg.model.beta}')
    lr = cfg.optimizer.lr
    constant_encoding = cfg.dataset.constant_encoding
    num_steps = cfg.dataset.num_step if constant_encoding else None
    u_th = cfg.model.u_th
    beta = cfg.model.beta
    surrgotate_dir = os.path.join(output_head, f'{dataset_name}_{model_name}_surrogate.txt')
    local_zo_dir = os.path.join(output_head, f'{dataset_name}_{model_name}_localzo.txt')
    with open(surrgotate_dir, 'a+') as f:
        f.write(f'u_th:{u_th},beta:{beta},num_steps:{num_steps},batch_size:{batch_size}\n')
    with open(local_zo_dir, 'a+') as f:
        f.write(f'u_th:{u_th},beta:{beta},num_steps:{num_steps},batch_size:{batch_size}\n')
    print(surrgotate_dir)
    print(local_zo_dir)
    set_output_dir(surrgotate_dir, local_zo_dir)
    print(f'num_steps:{num_steps}')
    train_loader = call(cfg.dataset.get_dataset)
    spconv_model, _ = call(cfg.model.get_models)
    optimizer = torch.optim.Adam(spconv_model.parameters(), lr=lr)
    max_iter = 5
    for i, (inputs, labels) in enumerate(train_loader):
        if i == max_iter:
            break

        if constant_encoding:
            inputs = repeat(inputs, num_steps)

        print('test input shape', inputs.shape)
        inputs = inputs.cuda()
        labels = labels.cuda()

        # forward and backward to see the sparsity of grad
        print('the sparsity of localzo')
        logits = spconv_model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()

        layer_index_reset()

    replace_leaky_zo_plain_once(spconv_model)

    for i, (inputs, labels) in enumerate(train_loader):
        if i == max_iter:
            break

        if constant_encoding:
            inputs = repeat(inputs, num_steps)

        inputs = inputs.cuda()
        labels = labels.cuda()

        print('the sparsity of surrogate')
        logits = spconv_model(inputs)
        loss = loss_fn(logits, labels)
        loss.backward()

        layer_index_reset()


if __name__ == '__main__':
    main()
