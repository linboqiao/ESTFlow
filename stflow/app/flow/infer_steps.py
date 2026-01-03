import os
import json
import wandb
import argparse
import numpy as np
import pandas as pd
from time import time, perf_counter
from tqdm import tqdm
from operator import itemgetter

import torch

from stflow.utils import set_random_seed, get_current_time, merge_fold_results
from stflow.data.dataset import HESTDatasetPath, MultiHESTDataset, padding_batcher, HESTDataset
from stflow.data.normalize_utils import get_normalize_method
from stflow.model.denoiser import Denoiser
from stflow.flow.interpolant import Interpolant
from stflow.app.flow.test import test
from stflow.hest_utils.utils import save_pkl

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, TensorDataset

# suppress all warnings
import warnings
warnings.filterwarnings("ignore")


dtype_to_bytes_linear = {"float32": 4, "float16": 2, "bfloat16": 2, "int8": 1, "int4": 0.5}

from torch import nn
def init_accelerator():
    print("可用GPU数量:", torch.cuda.device_count())
    torch.manual_seed(0)
    device_name = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    if device_name == "cpu":
        return

    device_module = getattr(torch, device_name, torch.cuda)
    device_module.reset_peak_memory_stats()
    device_module.manual_seed_all(0)
    # might not be necessary, but just to be sure
    nn.Linear(1, 1).to(device_name)


def strip_module_prefix(state):
    from collections import OrderedDict
    new_state = OrderedDict()
    for k, v in state.items():
        nk = k[7:] if k.startswith("module.") else k
        new_state[nk] = v
    return new_state


def main(args, split_id, train_sample_ids, test_sample_ids, val_save_dir, checkpoint_save_dir, checkpoint_load_dir=None):
    normalize_method = get_normalize_method(args.normalize_method)

    print("Dataset Loading")
    sample_id_paths = [
        HESTDatasetPath(
            name=sample_id,
            h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
            h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
            gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list),
        ) for sample_id in train_sample_ids
    ]
    # using the same train sample ids for validation
    sample_id_paths = [
        HESTDatasetPath(
                name=sample_id,
                h5_path=os.path.join(args.embed_dataroot, args.dataset, args.feature_encoder, f"fp32/{sample_id}.h5"),
                h5ad_path=os.path.join(args.source_dataroot, args.dataset, f"adata/{sample_id}.h5ad"),
                gene_list_path=os.path.join(args.source_dataroot, args.dataset, args.gene_list
            ),
        ) for sample_id in test_sample_ids
    ]
    val_loaders = [
        torch.utils.data.DataLoader(
            HESTDataset(
                sample_id_path, distribution="constant_1.0", 
                normalize_method=normalize_method,
                sample_times=1
            ),
            batch_size=1, collate_fn=padding_batcher()
        ) for sample_id_path in sample_id_paths
    ]

    model = Denoiser(args)

    diffusier = Interpolant(
        args.prior_sampler, 
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )
    
    device = args.device
    model = model.to(device)

    # 2. 加载权重
    if checkpoint_load_dir != None:
        checkpoint_path = os.path.join(checkpoint_load_dir, "pearson_best.pth")
        state = torch.load(checkpoint_path, map_location="cpu")  # 或 "cuda"
        state = strip_module_prefix(state)
        model.load_state_dict(state, strict=True)

    print("Testing")
    tic_total = perf_counter()

    max_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 16, 32, 64]
    
    all_best, best_val_dict = -1, []    

    for steps in max_steps:
        args.n_sample_steps = steps
        start = time()
        val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)
        end = time()
        time_sec = end - start
        best_pearson = val_perf_dict["all"]['pearson_mean']
        if best_pearson>all_best:
            best_val_dict = val_perf_dict
        for patch_name, dataset_res in val_perf_dict.items():
             with open(os.path.join(val_save_dir, f'{patch_name}_results_steps_{steps}_pearson_{best_pearson}_timesec_{time_sec}.json'), 'w') as f:
                  json.dump(dataset_res, f, sort_keys=True, indent=4)
        print(f"steps: {steps}, \tpearson_mean: {best_pearson}")

    return best_val_dict["all"]


def run(args):
    # get train/test splits
    split_dir = os.path.join(args.source_dataroot, args.dataset, 'splits')
    splits = os.listdir(split_dir)
    all_split_results = []
    
    for i in range(len(splits) // 2):  # each split has a train and test file so we divide by 2
        print(f"Running dataset {args.dataset} split {i}")

        train_df = pd.read_csv(os.path.join(split_dir, f'train_{i}.csv'))
        test_df = pd.read_csv(os.path.join(split_dir, f'test_{i}.csv'))

        train_sample_ids = train_df['sample_id'].tolist()
        test_sample_ids = test_df['sample_id'].tolist()

        kfold_save_dir = os.path.join(args.save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)        
        checkpoint_save_dir = os.path.join(kfold_save_dir, 'checkpoints')
        os.makedirs(checkpoint_save_dir, exist_ok=True)
        if args.load_dir is not None:
            kfold_load_dir = os.path.join(args.load_dir, f'split{i}')
            checkpoint_load_dir = os.path.join(kfold_load_dir, 'checkpoints')
        else:
            checkpoint_load_dir = None

        results = main(args, i, train_sample_ids, test_sample_ids, kfold_save_dir, checkpoint_save_dir, checkpoint_load_dir)
        all_split_results.append(results)

    kfold_results = merge_fold_results(all_split_results)
    with open(os.path.join(args.save_dir, f'results_kfold_best_step.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)

from pprint import pprint
import gc
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', default=["all"], help="all// LUNG, READ, HCC")
    parser.add_argument('--use_wandb', default=False)
    parser.add_argument('--source_dataroot', default="/path/to/data/")
    parser.add_argument('--embed_dataroot', type=str, default="/path/to/embed_data/")
    parser.add_argument('--gene_list', type=str, default='var_50genes.json')
    parser.add_argument('--save_dir', type=str, default=None, help="checkpoint_dir_save")
    parser.add_argument('--load_dir', type=str, default=None, help="checkpoint_dir_load")
    parser.add_argument('--feature_encoder', type=str, default='uni_v1_official', help="uni_v1_official | resnet50_trunc | ciga | gigapath")
    parser.add_argument('--normalize_method', type=str, default="log1p")    
    parser.add_argument('--exp_code', type=str, default="multiview")
    parser.add_argument('--multiview', default=True)

    # training hyperparameters
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--sample_times', type=int, default=10, help='Number of times to sample patches from each image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--early_stop_step', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--eval_step', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataloader')
    parser.add_argument('--loss_func', type=str, default='mse', help="mse | mae | pearson")
    parser.add_argument('--patch_distribution', type=str, default='uniform')
    parser.add_argument('--n_genes', type=int, default=50)

    # flow matching hyperparameters
    parser.add_argument('--n_sample_steps', type=int, default=32)
    parser.add_argument('--prior_sampler', type=str, default="zinb", help="gaussian | uniform | zero | zinb")
    parser.add_argument('--zinb_logits', type=float, default=0.1)
    parser.add_argument('--zinb_total_count', type=float, default=1)
    parser.add_argument('--zinb_zi_logits', type=float, default=0., help="Prob for zero inflation")  # before sigmoid

    # model hyperparameters
    parser.add_argument('--backbone', type=str, default="spatial_transformer")
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--pairwise_hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--attn_dropout', type=float, default=0.2)
    parser.add_argument('--n_neighbors', type=int, default=32)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=1024, help="uni:1024, ciga:512")
    parser.add_argument('--norm', type=str, default='layer', help="batch | layer")
    parser.add_argument('--activation', type=str, default='swiglu', help="relu | gelu | swiglu")
    args = parser.parse_args()

    args.distributed = False

    args.feature_dim = {
        "uni_v1_official": 1024,
        "gigapath": 1536,
        "ciga": 512,
    }[args.feature_encoder]

    set_random_seed(args.seed)

    if args.exp_code == "multiview":
        args.multiview = True

    if args.exp_code is None:
        exp_code = f"{args.backbone}::{get_current_time()}"
    else:
        exp_code = args.exp_code + f"_{args.feature_encoder}" + f"_{args.backbone}::{get_current_time()}"
        exp_code = args.exp_code + f"_{args.feature_encoder}" + f"_{args.backbone}"    
    
    save_dir = os.path.join(args.save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)
    load_dir = args.load_dir
    pprint(args)

    if args.datasets[0] == "all":
        args.datasets = ["IDC", "PRAD", "PAAD", "SKCM", "COAD", "READ", "CCRCC",  "HCC", "LUNG", "LYMPH_IDC"]

    for dataset in args.datasets:
        args.dataset = dataset
        args.save_dir = os.path.join(save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)
        if args.load_dir is not None:
            args.load_dir = os.path.join(load_dir, dataset)

        with open(os.path.join(args.save_dir, 'config_infer_steps.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)

        for i in range(1):
            run(args)
            torch.cuda.empty_cache()
            gc.collect()
