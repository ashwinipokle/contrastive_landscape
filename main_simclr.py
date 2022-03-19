import torch
import torch.nn as nn

from torch.utils.data import DataLoader

import numpy as np
from data_model.gen_sparse_coding_data import gen_z, gen_M, gen_Winit, gen_epsilon

from config import config
from pathlib import Path
import pickle

from common_args import parse_args

from dataset.simple_contrastive_dataset import SimpleContrastiveDataset
from dataset.masked_sparse_contr_dataset import MaskedSparseContrastiveDataset
from dataset.multimask_sparse_contr_dataset import MultiMaskedSparseContrastiveDataset, multi_mask_data_collate

from models.simclr import SimCLROrigModel, SimCLRModel
from models.simclr_asym import SimCLRAsymmetricModel

from functions.train import train, augment_and_train

import wandb

def main():
    args = parse_args()

    print('config.n ', config.nn)  # debug
    print('config.p ', config.p)  # debug
    print('config.d ', config.d)  # debug
    print('config.m ', config.m)  # debug
    print('config.has_target_predictor ', config.has_target_predictor)  # debug
    print('config.has_target_ReLU ', config.has_target_ReLU)  # debug
    print('config.lr ' , config.lr)  # debug
    print('config.sigma0 ', config.sigma0)  # debug
    print('args.normalize_repr', args.normalize_repr)
    print('args.ema_decay', args.ema_decay)
    print('args.use_multimasking', args.use_multimasking)
    print('args.use_bn', args.use_bn)
    print('args.temperature', args.temperature)
    print('args.use_pred', args.use_pred)
    print('args.m_identity', args.m_identity)

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda:{}'.format(args.local_rank))
    print(f"Using device {device}")

    # Save Result
    root_output_dir = Path(config.LOG_DIR)
    
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir(parents=True)

    log_metrics = args.log_metrics
    logger = None

    sigma0 = None
    # number of augmentations if multimask
    n_aug = args.n_aug

    for sigma0 in config.gaussian_noise_levels:
        for sparsity in config.sparsity_levels: # proportion of non-zeros
            for maskprob in config.masking_probs:
                for i in range(config.num_exp):
                    print(f"Experiment {i+1}")
                    
                    np.random.seed(i+1)
                    torch.manual_seed(i+1)

                    if log_metrics:
                        run = wandb.init(project=args.wandb_project, reinit=True, name=f"trial-{i}",
                                                group=f"{args.model}-I{args.m_identity}-bn-{args.use_bn}-norm-{args.normalize_repr}-p{config.p}-m{config.m}-d{config.d}-sp-{sparsity}-" + 
                                                    f"mask{maskprob}-lr-{config.lr}-1h-{config.one_hot_latent}",
                                                config=config)
                        logger = wandb.log

                    # Generate data
                    if args.m_identity:
                        M = np.eye(config.p)
                    else:
                        M = gen_M(p=config.p, d=config.d)

                    Z = gen_z(n=config.nn, d=config.d, prob=sparsity, one_hot_latent=config.one_hot_latent)

                    Epsilon = gen_epsilon(n=config.nn, p=config.p, d=config.d, sigma0 = sigma0)
                    X = (M @ Z + Epsilon).T

                    if args.use_multimasking:
                        dataset = MultiMaskedSparseContrastiveDataset(data=X, Z=Z.T, prob_ones=1-maskprob, n_aug=n_aug)
                        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=multi_mask_data_collate)
                    else:
                        dataset = MaskedSparseContrastiveDataset(data=X, Z=Z.T, prob_ones=1-maskprob)
                        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

                    Wo_init = gen_Winit(M, c=None, m=config.m, d=config.d, p=config.p)

                    if args.model == 'simclr-orig':
                        # drop the last incomplete batch
                        # Note: Use a larger dataset to train this, ensure that the model can see most of the dataset
                        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

                        model = SimCLROrigModel(Wo_init=Wo_init,
                                                m=config.m,
                                                p=config.p,
                                                d=config.d,
                                                has_online_ReLU=config.has_online_ReLU,
                                                has_target_ReLU = config.has_target_ReLU,
                                                batch_size=config.batch_size,
                                                temperature=args.temperature,
                                                device=device)

                        model = model.to(device)

                        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
                        val_dict = train(model, optimizer=optimizer,
                                        train_loader=train_loader,
                                        max_epochs=config.NUM_EPOCHES,
                                        M=M,
                                        log_metrics=log_metrics,
                                        logger=logger
                                        )
                    elif args.model == 'simclr':
                            model = SimCLRModel(Wo_init=Wo_init,
                                                    m=config.m,
                                                    p=config.p,
                                                    d=config.d,
                                                    has_online_ReLU=config.has_online_ReLU,
                                                    has_target_ReLU = config.has_target_ReLU,
                                                    batch_size=config.batch_size * n_aug if args.use_multimasking else config.batch_size,
                                                    temperature=args.temperature,
                                                    use_bn=args.use_bn,
                                                    device=device)

                            model = model.to(device)

                            optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
                            val_dict = train(model, optimizer=optimizer,
                                            train_loader=train_loader,
                                            max_epochs=config.NUM_EPOCHES,
                                            M=M,
                                            log_metrics=log_metrics,
                                            logger=logger
                                            )

                    elif args.model == 'simclr-alter-aug':
                        dataset = SimpleContrastiveDataset(data=X, Z=Z.T, prob_ones=1-maskprob)
                        train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=True)

                        model = SimCLRAsymmetricModel(Wo_init=Wo_init,
                                                m=config.m,
                                                p=config.p,
                                                d=config.d,
                                                has_online_ReLU=config.has_online_ReLU,
                                                has_target_ReLU = config.has_target_ReLU,
                                                batch_size=config.batch_size,
                                                temperature=args.temperature,
                                                device=device)

                        model = model.to(device)

                        optimizer = torch.optim.SGD(model.parameters(), lr=config.lr)
                        val_dict = augment_and_train(model, optimizer=optimizer,
                                        train_loader=train_loader,
                                        max_epochs=config.NUM_EPOCHES,
                                        M=M,
                                        prob_ones=1-maskprob,
                                        log_metrics=log_metrics,
                                        logger=logger
                                        )

                    if log_metrics:
                        run.finish()

                    with open(root_output_dir.joinpath(f"training_val_dict_c{None}_noise{sigma0}_sparse{sparsity}_mask{maskprob}_ema{args.ema_decay}_experiment{i}.pkl"), 'wb') as f:
                        pickle.dump(val_dict, f)

                    torch.save({
                            'model_state_dict': model.state_dict()
                        }, root_output_dir.joinpath('final_model.pt'))

if __name__ == '__main__':
    main()
