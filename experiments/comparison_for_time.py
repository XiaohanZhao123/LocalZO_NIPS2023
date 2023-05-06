gpu_idx = '3'
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_idx

import torch
from utils import train_and_profile_snntorch, train_and_profile_spconv
from snntorch import functional as SF
import hydra
from hydra.utils import call


loss_fn = SF.mse_count_loss()
acc_fn = SF.accuracy_rate


@hydra.main(config_path='config', config_name='defaults', version_base=None)
def main(cfg):
    print(cfg)
    lr = cfg.optimizer.lr
    constant_encoding = cfg.dataset.constant_encoding
    num_steps = cfg.dataset.num_step if constant_encoding else None
    train_loader = call(cfg.dataset.get_dataset)
    spconv_model, snntorch_model = call(cfg.model.get_models)
    # train_and_profile_spconv(net=spconv_model,
    #                          train_loader=train_loader,
    #                          num_epochs=2,
    #                          lr=lr,
    #                          constant_encoding=constant_encoding,
    #                          num_steps=num_steps,
    #                          save_dir=None)
    torch.cuda.empty_cache()
    train_and_profile_snntorch(net=snntorch_model,
                               train_loader=train_loader,
                               num_epochs=2,
                               lr=lr,
                               constant_encoding=constant_encoding,
                               num_steps=num_steps)


if __name__ == '__main__':
    main()
