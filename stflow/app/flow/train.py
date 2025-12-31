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

from stflow.utils import set_random_seed, get_current_time, merge_fold_results, init_distributed_mode
from stflow.data.dataset import HESTDatasetPath, MultiHESTDataset, padding_batcher, HESTDataset
from stflow.data.normalize_utils import get_normalize_method
from stflow.model.denoiser import Denoiser
from stflow.flow.interpolant import Interpolant
from stflow.app.flow.test import test
from stflow.hest_utils.utils import save_pkl
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as NativeDDP
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
    train_dataset = MultiHESTDataset(sample_id_paths,
                                     distribution=args.patch_distribution, 
                                     normalize_method=normalize_method,
                                     sample_times=args.sample_times)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=padding_batcher(),sampler=train_sampler)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=padding_batcher())

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
    if args.local_rank < 0:
        model = model.cuda()
    else:
        model = model.cuda(args.local_rank)
    
    '''
    modules_linear = [n for n, m in model.named_modules() if "linear" in type(m).__name__.lower()]
    from peft import LoraConfig
    config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules = modules_linear[:-4],            
        modules_to_save = modules_linear[-4:]
        )
    from peft import get_peft_model
    model = get_peft_model(model, config)
    '''

    if args.distributed:
        if args.rank == 0:
            print("Using native Torch DistributedDataParallel.")
            pprint(args)
        model = NativeDDP(model, device_ids=[args.local_rank])
        dist.barrier()
    else:
        device = args.device
        model = model.to(device)

    diffusier = Interpolant(
        args.prior_sampler, 
        total_count=torch.tensor([args.zinb_total_count]),
        logits=torch.tensor([args.zinb_logits]),
        zi_logits=args.zinb_zi_logits,
        normalize=args.prior_sampler != "gaussian",
    )

    
    # 2. 加载权重
    checkpoint_path = os.path.join(checkpoint_load_dir, "pearson_best.pth")
    if os.path.exists(checkpoint_path):
        map_location=f"cuda:{args.local_rank}"
        state_dict = torch.load(checkpoint_path, map_location="cpu")  # 或 "cpu"

        '''
        # 如果checkpoint是DDP保存的, 可能需要移除'module.'前缀
        if not isinstance(model, NativeDDP) and list(state_dict.keys())[0].startswith('module.'):
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        
        # 如果当前是DDP但checkpoint不是, 添加'module.'前缀
        if isinstance(model, NativeDDP) and not list(state_dict.keys())[0].startswith('module.'):
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
        print(state_dict.keys())
        '''
        if isinstance(model, NativeDDP):
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

        val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)
        best_pearson = val_perf_dict["all"]['pearson_mean']
        if args.local_rank==0:
            for patch_name, dataset_res in val_perf_dict.items():
                with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                    json.dump(dataset_res, f, sort_keys=True, indent=4)
                with open(os.path.join(val_save_dir, f'{patch_name}_results_pearson_{best_pearson:.6f}.json'), 'w') as f:
                    json.dump(dataset_res, f, sort_keys=True, indent=4)

    '''
    init_accelerator()
    device_name = torch.accelerator.current_accelerator().type if hasattr(torch, "accelerator") else "cuda"
    device_module = getattr(torch, device_name, torch.cuda)
    accelerator_memory_init = device_module.max_memory_allocated()
    accelerator_memory_log = []
    '''
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print("Training")
    tic_total = perf_counter()

    best_pearson, best_val_dict = -1, None
    early_stop_step = 0
    if args.local_rank == 0:
        epoch_iter = tqdm(range(1, args.epochs + 1), ncols=100)
    else:
        epoch_iter = range(1, args.epochs + 1)

    for epoch in epoch_iter:
        avg_loss = 0
        model.train()

        for step, batch in enumerate(train_loader):
            batch = [x.to(args.device) for x in batch]
            img_features, coords, gene_exp = batch

            noisy_exp, t_steps = diffusier.corrupt_exp(gene_exp)
            pred_exp, loss = model(
                exp=noisy_exp,
                img_features=img_features,
                coords=coords,
                labels=gene_exp,
                t_steps=t_steps
            )

            optimizer.zero_grad()
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            optimizer.step()

            '''
            mem_train = device_module.memory_allocated() - accelerator_memory_init
            accelerator_memory_log.append(mem_train)
            print(f"memory train: {mem_train // 2**20}MB")
            '''

            avg_loss += loss.cpu().item()
        
        avg_loss /= len(train_loader)
        if args.local_rank == 0:
            epoch_iter.set_description(f"epoch: {epoch}, avg_loss: {avg_loss:.3f}, best_pearson: {best_pearson:.3f}")

        if epoch % args.eval_step == 0 or epoch == args.epochs:
            if args.local_rank == 0:
                val_perf_dict, pred_dump = test(args, diffusier, model, val_loaders, return_all=True)
                if val_perf_dict["all"]['pearson_mean'] > best_pearson:
                    best_pearson = val_perf_dict["all"]['pearson_mean']
                    best_val_dict = val_perf_dict
                    for patch_name, dataset_res in val_perf_dict.items():
                        with open(os.path.join(val_save_dir, f'{patch_name}_results.json'), 'w') as f:
                            json.dump(dataset_res, f, sort_keys=True, indent=4)
                        with open(os.path.join(val_save_dir, f'{patch_name}_results_pearson_{best_pearson:.6f}.json'), 'w') as f:
                            json.dump(dataset_res, f, sort_keys=True, indent=4)
                            
                    torch.save(model.module.state_dict(), os.path.join(checkpoint_save_dir, f"pearson_{best_pearson:.6f}.pth"))
                    torch.save(model.module.state_dict(), os.path.join(checkpoint_save_dir, "pearson_best.pth"))
                    
                    early_stop_step = 0

            else:
                early_stop_step += 1
                if early_stop_step >= args.early_stop_step:
                    print(f"pearson_mean: {best_pearson}")
                    print("Early stopping")
                    break


    '''
    toc_total = perf_counter()
    accelerator_memory_final = device_module.max_memory_allocated()
    accelerator_memory_avg = int(sum(accelerator_memory_log) / len(accelerator_memory_log))
    print(f"memory avg: {accelerator_memory_avg // 2**20}MB")
    print(f"memory max: {(accelerator_memory_final - accelerator_memory_init) // 2**20}MB")
    print(f"total time: {toc_total - tic_total:.2f}s")
    '''
    if args.distributed:
        dist.barrier()
    if args.local_rank == 0:
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
    with open(os.path.join(args.save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)

from pprint import pprint
import gc
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--datasets', nargs='+', default=["IDC"], help="all// LUNG, READ, HCC")
    parser.add_argument('--source_dataroot', default="/nas/linbo/biospace/exps/20240112-His2ST/ESTFlow/dataset/hest-bench/")
    parser.add_argument('--embed_dataroot', type=str, default="/nas/linbo/biospace/exps/20240112-His2ST/ESTFlow/embed_data/hest-bench/")
    parser.add_argument('--gene_list', type=str, default='var_50genes.json')
    parser.add_argument('--save_dir', type=str, default="/nas/linbo/biospace/exps/20240112-His2ST/ESTFlow/results_dir/hest-bench/")
    parser.add_argument('--load_dir', type=str, default="/nas/linbo/biospace/exps/20240112-His2ST/ESTFlow/results_dir/hest-bench/multiview_uni_v1_official_spatial_transformer", help="checkpoint_dir_load")
    parser.add_argument('--feature_encoder', type=str, default='uni_v1_official', help="uni_v1_official | resnet50_trunc | ciga | gigapath")
    parser.add_argument('--normalize_method', type=str, default="log1p")
    parser.add_argument('--exp_code', type=str, default="multiview")
    #parser.add_argument('--multiview', default=False)

    # training hyperparameters
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--sample_times', type=int, default=10, help='Number of times to sample patches from each image')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--early_stop_step', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=45)
    parser.add_argument('--clip_norm', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--eval_step', type=int, default=5)
    parser.add_argument('--num_workers', type=int, default=16, help='Number of workers for dataloader')
    parser.add_argument('--loss_func', type=str, default='mse', help="mse | mae | pearson")
    parser.add_argument('--patch_distribution', type=str, default='uniform')
    parser.add_argument('--n_genes', type=int, default=50)    

    # flow matching hyperparameters
    parser.add_argument('--n_sample_steps', type=int, default=5)
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
    parser.add_argument('--n_neighbors', type=int, default=16)
    parser.add_argument('--n_heads', type=int, default=4)
    parser.add_argument('--feature_dim', type=int, default=1024, help="uni:1024, ciga:512")
    parser.add_argument('--norm', type=str, default='layer', help="batch | layer")
    parser.add_argument('--activation', type=str, default='swiglu', help="relu | gelu | swiglu")
    parser.add_argument("--local-rank", default=-1, type=int)
    parser.add_argument('--gpus', type=int,
                    help='number of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--dist-backend', type=str, default='nccl')
    parser.add_argument('--port', type=int, default='12345')
    args = parser.parse_args()

    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.distributed = int(os.environ['WORLD_SIZE']) > 1
    if args.distributed:
        init_distributed_mode(args)
        dist.barrier()
        print('Training in distributed mode with multiple processes, 1 GPU per process. Process %d, total %d.'
                     % (args.rank, args.world_size))
    else:
        print('Training with a single process on 1 GPUs.')

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

    if args.datasets[0] == "all":
        args.datasets = ["SKCM", "PAAD", "PRAD", "IDC", "READ", "LUNG", "HCC", "COAD", "LYMPH_IDC", "CCRCC"]

    for dataset in args.datasets:
        args.dataset = dataset
        args.save_dir = os.path.join(save_dir, dataset)
        os.makedirs(args.save_dir, exist_ok=True)
        if args.load_dir is not None:
            args.load_dir = os.path.join(load_dir, dataset)

        with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, sort_keys=True, indent=4)
        
        run(args)
        torch.cuda.empty_cache()
        gc.collect()
