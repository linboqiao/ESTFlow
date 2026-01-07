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
        args.distributed = True
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
    
    print("GPU:", args.gpu, "device:", args.device)
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'    
    if args.dist_url == "env://":
        master_addr = os.environ.get('MASTER_ADDR', 'localhost')
        master_port = os.environ.get('MASTER_PORT', '29500')
        args.port = master_port
        args.dist_url = f"tcp://{master_addr}:{master_port}"
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank,
                                         timeout=datetime.timedelta(seconds=600))
    setup_for_distributed(args.rank == 0)

import gc
def cleanup_distributed():
    """Clean up distributed training"""
    dist.destroy_process_group()    
    torch.cuda.empty_cache()
    gc.collect()
    
def print_env_vars():
    import socket
    """打印所有相关环境变量"""
    print("=" * 50)
    print("Hostname:", socket.gethostname())
    print("=" * 50)
    
    # torchrun 相关变量
    torchrun_vars = [
        'MASTER_ADDR', 'MASTER_PORT',
        'WORLD_SIZE', 'RANK', 'LOCAL_RANK',
        'LOCAL_WORLD_SIZE', 'NODE_RANK',
        'TORCHELASTIC_RUN_ID'
    ]
    
    for var in torchrun_vars:
        value = os.environ.get(var, 'NOT SET')
        print(f"{var}: {value}")
    
    # CUDA 信息
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'NOT SET')}")
    print(f"Available GPUs: {torch.cuda.device_count()}")    


def comprehensive_communication_test():
    """综合通信测试"""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    print(f"\n{'='*50}")
    print(f"Rank {rank}/{world_size}: 开始通信健康检查")
    print(f"{'='*50}")
    
    tests_passed = 0
    total_tests = 5
    
    try:
        # 测试1: 检查是否已初始化
        if dist.is_initialized():
            print(f"✓ 测试1: 分布式已初始化 (rank={rank}, world_size={world_size})")
            tests_passed += 1
        else:
            print(f"✗ 测试1: 分布式未初始化")
            return False
        
        # 测试2: 屏障同步
        dist.barrier()
        print(f"✓ 测试2: 屏障同步正常")
        tests_passed += 1
        
        # 测试3: 广播测试
        test_tensor = torch.tensor([rank], dtype=torch.float32).cuda()
        dist.broadcast(test_tensor, src=0)
        
        expected = rank if rank != 0 else 0.0  # 广播后所有rank的值应该相同
        if torch.allclose(test_tensor, torch.tensor([expected]).cuda()):
            print(f"✓ 测试3: 广播测试正常 (值={test_tensor.item():.1f})")
            tests_passed += 1
        else:
            print(f"✗ 测试3: 广播测试失败")
        
        # 测试4: all_reduce 测试
        reduce_tensor = torch.tensor([1.0], dtype=torch.float32).cuda(rank)
        dist.all_reduce(reduce_tensor, op=dist.ReduceOp.SUM)

        print(f"Rank {rank}: all_reduce SUM 后 = {reduce_tensor}")
        
        if torch.allclose(reduce_tensor, torch.tensor([world_size]).cuda()):
            print(f"✓ 测试4: all_reduce SUM 正常 (结果={reduce_tensor.item():.1f})")
            tests_passed += 1
        else:
            print(f"✗ 测试4: all_reduce 测试失败")
        
        # 测试5: all_gather 测试
        local_data = torch.tensor([float(rank)], dtype=torch.float32).cuda()
        gathered = [torch.zeros_like(local_data) for _ in range(world_size)]
        dist.all_gather(gathered, local_data)
        
        # 验证收集的数据
        correct = True
        for i, t in enumerate(gathered):
            if not torch.allclose(t, torch.tensor([float(i)]).cuda()):
                correct = False
                break
        
        if correct:
            print(f"✓ 测试5: all_gather 正常")
            tests_passed += 1
        else:
            print(f"✗ 测试5: all_gather 测试失败")
        
        # 最终同步
        dist.barrier()
        
        # 汇总结果
        success_rate = tests_passed / total_tests * 100
        print(f"\nRank {rank}: 通信测试完成 - {tests_passed}/{total_tests} 通过 ({success_rate:.1f}%)")
        
        return tests_passed == total_tests
        
    except Exception as e:
        print(f"✗ 通信测试异常: {e}")
        import traceback
        traceback.print_exc()
        return False
    


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
