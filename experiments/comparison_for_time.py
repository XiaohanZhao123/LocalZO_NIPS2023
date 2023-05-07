gpu_idx = '3'
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

import torch
from utils import train_and_profile_snntorch, train_and_profile_spconv
import hydra
from hydra.utils import call


@hydra.main(config_path='config', config_name='defaults', version_base=None)
def main(cfg):
    dataset_name = cfg.dataset.name
    model_name = cfg.model.name
    print(f'testing on {model_name}, {dataset_name}')
    print(f'model setup: u_th:{cfg.model.u_th}, beta:{cfg.model.beta}')
    lr = cfg.optimizer.lr
    constant_encoding = cfg.dataset.constant_encoding
    num_steps = cfg.dataset.num_step if constant_encoding else None
    train_loader = call(cfg.dataset.get_dataset)
    spconv_model, snntorch_model = call(cfg.model.get_models)
    spconv_mean_time = train_and_profile_spconv(net=spconv_model,
                                                train_loader=train_loader,
                                                num_epochs=2,
                                                lr=lr,
                                                constant_encoding=constant_encoding,
                                                num_steps=num_steps,
                                                save_dir=None)

    torch.cuda.empty_cache()
    snntorch_mean_time = train_and_profile_snntorch(net=snntorch_model,
                                                    train_loader=train_loader,
                                                    num_epochs=2,
                                                    lr=lr,
                                                    constant_encoding=constant_encoding,
                                                    num_steps=num_steps)
    with open(f'/home/zxh/remote/sparse_zo_nips2023/experiments/results/{dataset_name}_{model_name}_snntorch_time.txt',
              'a+') as f:
        f.write(
            f'batch_size:{cfg.dataset.batch_size}, num_step:{num_steps}, snn_torch_mean_time:{snntorch_mean_time}, spconv_mean_time{spconv_mean_time}\n')
    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
