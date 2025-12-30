import torch
import numpy as np
import random
import mygene
import datetime
import os
import subprocess
import torch.nn as nn
import torch.distributed as dist
import ctypes

def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        args.dist_url = 'env://'
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.device = 'cuda:%d' % args.gpu
        os.environ['LOCAL_SIZE'] = str(torch.cuda.device_count())
    elif 'SLURM_PROCID' in os.environ:
        print("SLURM PROCID here")
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
        node_list = os.environ['SLURM_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '29501')  # 29500
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)
        args.local_rank = proc_id % num_gpus
        args.dist_url = 'env://'
        args.world_size = ntasks
        args.rank = proc_id
        args.gpu = proc_id % num_gpus   # local rank
        args.device = 'cuda:%d' % args.gpu
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    print("gpu", args.gpu)
    print("device", args.device)
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)
    
def symbol2ensembl_id(gene_symbol, species='human'):
    try:
        mg = mygene.MyGeneInfo()
        gene_info = mg.query(gene_symbol, species=species, fields='ensembl.gene')
        if gene_info['total'] == 0:
            return None
        return gene_info['hits'][0].get('ensembl')['gene']
    except Exception as e:
        return None


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def merge_fold_results(arr):
    aggr_dict = {}
    for dict in arr:
        for item in dict['pearson_corrs']:
            gene_name = item['name']
            correlation = item['pearson_corr']
            aggr_dict[gene_name] = aggr_dict.get(gene_name, []) + [correlation]
    
    aggr_results = []
    all_corrs = []
    for key, value in aggr_dict.items():
        aggr_results.append({
            "name": key,
            "pearson_corrs": value,
            "mean": np.mean(value),
            "std": np.std(value)
        })
        all_corrs += value
        
    mean_per_split = [d['pearson_mean'] for d in arr]    
        
    return {"pearson_corrs": aggr_results, "pearson_mean": np.mean(mean_per_split), "pearson_std": np.std(mean_per_split), "mean_per_split": mean_per_split}


def get_current_time():
    now = datetime.datetime.now()
    year = now.year % 100  # convert to 2-digit year
    month = now.month
    day = now.day
    hour = now.hour
    minute = now.minute
    second = now.second
    return f"{year:02d}-{month:02d}-{day:02d}-{hour:02d}-{minute:02d}-{second:02d}"


if __name__ == "__main__":
    print(symbol2ensembl_id("CDK1"))
    print(symbol2ensembl_id("0610005C13Rik", species='mouse'))
