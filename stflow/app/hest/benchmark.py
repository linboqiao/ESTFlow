import os
import json
import argparse
from tqdm import tqdm
from operator import itemgetter
import numpy as np
import pandas as pd

import torch
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import StandardScaler
from huggingface_hub import snapshot_download, hf_hub_download

from trainer import train_test_reg
from stflow.utils import set_random_seed
from stflow.hest_utils.encoder import load_encoder
from stflow.hest_utils.st_dataset import H5TileDataset, load_adata
from stflow.hest_utils.utils import get_current_time, save_pkl, merge_dict, get_path
from stflow.hest_utils.file_utils import save_hdf5, read_assets_from_h5
from stflow.data.normalize_utils import get_normalize_method


class LazyEncoder:
    def __init__(self, name, weights_root, private_weights_root=None, transforms=None, model=None):
        self.name = name
        self.model = model
        self.transforms = transforms
        self.weights_root = weights_root
        self.private_weights_root = private_weights_root
        
    def get_model(self, device):
        if self.model is not None:
            return self.model, self.transforms
        else:
            encoder, img_transforms, _ = load_encoder(self.name, device, self.weights_root, self.private_weights_root)
            return encoder, img_transforms


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


def embed_tiles(dataloader,
                model,
                embedding_save_path,
                device,
                precision=torch.float32,
                use_coords=None):
    
    def post_collate_fn(batch):
        """
        Post collate function to clean up batch
        """
        if batch["imgs"].dim() == 5:
            assert batch["imgs"].size(0) == 1
            batch["imgs"] = batch["imgs"].squeeze(0)
        if batch["coords"].dim() == 3:
            assert batch["coords"].size(0) == 1
            batch["coords"] = batch["coords"].squeeze(0)
        return batch

    """
    Extract embeddings from tiles using encoder and save to h5 file
    """
    model.eval()
    for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc='Embedding Tiles', ncols=100):
        batch = post_collate_fn(batch)
        imgs = batch['imgs'].to(device)
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=precision):
            if use_coords:
                embeddings = model(imgs, batch['coords'].to(device))
            else:
                embeddings = model(imgs)
        if batch_idx == 0:
            mode = 'w'
        else:
            mode = 'a'
        asset_dict = {'embeddings': embeddings.cpu().numpy()}
        asset_dict.update({key: np.array(val) for key, val in batch.items() if key != 'imgs'})
        save_hdf5(embedding_save_path,
                  asset_dict=asset_dict,
                  mode=mode)
    return embedding_save_path  


def predict_single_split(train_split, test_split, args, save_dir, dataset_name, lazy_enc, device, precision, source_dataroot):
    if not os.path.isfile(train_split):
        train_split = os.path.join(source_dataroot, 'splits', train_split)
    if not os.path.isfile(test_split):
        test_split = os.path.join(source_dataroot, 'splits', test_split)
    
    train_df = pd.read_csv(train_split)
    test_df = pd.read_csv(test_split)
    
    embedding_dir = os.path.join(get_path(args.embed_dataroot), dataset_name, lazy_enc.name, args.precision)
    os.makedirs(embedding_dir, exist_ok=True)
    
    # perform embedding
    print(f"Embedding tiles using {lazy_enc.name} encoder")
    encoder, img_transforms = lazy_enc.get_model(device)

    for split in [train_df, test_df]:
        for i in tqdm(range(len(split)), desc=f'Embedding {split}', ncols=100):
            sample_id = split.iloc[i]['sample_id']
            tile_h5_path = os.path.join(source_dataroot, split.iloc[i]['patches_path'])
            assert os.path.isfile(tile_h5_path), f"Tile file {tile_h5_path} does not exist"
            embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
            if not os.path.isfile(embed_path) or args.overwrite:
                #if lazy_enc.model is None:
                #    encoder, img_transforms, enc_config = load_encoder(lazy_enc, device) # delay instantiation incase we do not need to perform embedding
                tile_dataset = H5TileDataset(tile_h5_path, chunk_size=args.batch_size, img_transform=img_transforms)
                tile_dataloader = torch.utils.data.DataLoader(tile_dataset, 
                                                          batch_size=1, 
                                                          shuffle=False,
                                                          num_workers=args.num_workers)
                _ = embed_tiles(tile_dataloader, encoder, embed_path, device, precision=precision, use_coords=(lazy_enc.name=="gigapathslide"))
            else:
                print(f"Skipping {sample_id} as it already exists")

    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    #### Linear Probe Evaluation ####
    print(f"Linear Probe Evaluation with Sklearn")
    
    # load and gather all data
    all_split_assets = {}
    
    gene_list = args.gene_list

    print(f'using gene_list {gene_list}')
    with open(os.path.join(source_dataroot, gene_list), 'r') as f:
        genes = json.load(f)['genes']

    for split_key, split in zip(['train', 'test'], [train_df, test_df]):
        split_assets = {}
        for i in tqdm(range(len(split)), desc=f'Loading {split_key} split', ncols=100):
            sample_id = split.iloc[i]['sample_id']
            embed_path = os.path.join(embedding_dir, f'{sample_id}.h5')
            expr_path = os.path.join(source_dataroot, split.iloc[i]['expr_path'])
            assets, attrs = read_assets_from_h5(embed_path)
            barcodes = assets['barcodes'].flatten().astype(str).tolist()
            adata = load_adata(expr_path, genes=genes, barcodes=barcodes, normalize_method=get_normalize_method('log1p'))
            assets['adata'] = adata.values
            split_assets = merge_dict(split_assets, assets)
        for key, val in split_assets.items(): 
            split_assets[key] = np.concatenate(val, axis=0)

        all_split_assets[split_key] = split_assets  
        print(f"Loaded {split_key} split with {len(split_assets['embeddings'])} samples: {split_assets['embeddings'].shape}")
    
    X_train, y_train = all_split_assets['train']['embeddings'], all_split_assets['train']['adata']
    X_test, y_test = all_split_assets['test']['embeddings'], all_split_assets['test']['adata']

    if args.dimreduce == 'PCA':
        print('perform PCA dim reduction')
        pipe = Pipeline([('scaler', StandardScaler()), (f'{args.dimreduce}', eval(args.dimreduce)(n_components=args.latent_dim))])
        X_train, X_test = torch.Tensor(pipe.fit_transform(X_train)), torch.Tensor(pipe.transform(X_test))
    
    linprobe_results, linprobe_dump = train_test_reg(X_train, X_test, y_train, y_test, random_state=args.seed, genes=genes, method=args.method)
    linprobe_summary = {}
    linprobe_summary.update({'n_train': len(y_train), 'n_test': len(y_test)})
    linprobe_summary.update({key: val for key, val in linprobe_results.items()})
    print(linprobe_summary)

    with open(os.path.join(save_dir, f'results.json'), 'w') as f:
        json.dump(linprobe_results, f, sort_keys=True, indent=4)
    with open(os.path.join(save_dir, f'summary.json'), 'w') as f:
        json.dump(linprobe_summary, f, sort_keys=True, indent=4)
    save_pkl(os.path.join(save_dir, f'inference_dump.pkl'), linprobe_dump)

    return linprobe_results


def predict_folds(args, exp_save_dir, enc, dataset_name, device, precision, source_dataroot):
    split_dir = os.path.join(source_dataroot, 'splits')
    splits = os.listdir(split_dir)
    n_splits = len(splits) // 2
    
    libprobe_results_arr = []
    for i in range(n_splits):
        print(f"Running split {i}")
        train_split = os.path.join(split_dir, f'train_{i}.csv')
        test_split = os.path.join(split_dir, f'test_{i}.csv')
        kfold_save_dir = os.path.join(exp_save_dir, f'split{i}')
        os.makedirs(kfold_save_dir, exist_ok=True)
        linprobe_results = predict_single_split(train_split, test_split, args, kfold_save_dir, dataset_name, lazy_enc=enc, device=device, precision=precision, source_dataroot=source_dataroot)
        libprobe_results_arr.append(linprobe_results)
    
    kfold_results = merge_fold_results(libprobe_results_arr)
    with open(os.path.join(exp_save_dir, f'results_kfold.json'), 'w') as f:
        p_corrs = kfold_results['pearson_corrs']
        p_corrs = sorted(p_corrs, key=itemgetter('mean'), reverse=True)
        kfold_results['pearson_corrs'] = p_corrs
        json.dump(kfold_results, f, sort_keys=True, indent=4)
        
    return kfold_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
    parser.add_argument('--overwrite', default=False, help='overwrite existing results')
    parser.add_argument('--source_dataroot', default="/home/username/Anonymous_STFlow/dataset/")
    parser.add_argument('--embed_dataroot', type=str, default="/home/username/Anonymous_STFlow/dataset/embed_dataroot")
    parser.add_argument('--weights_root', type=str, default="/home/username/Anonymous_STFlow/dataset/weights_root")
    parser.add_argument('--results_dir', type=str, default="home/username/Anonymous_STFlow/results_dir/hest")
    parser.add_argument('--exp_code', type=str, default="uni")
    
    parser.add_argument('--precision', type=str, default='fp32', help='Precision (fp32 or fp16)')
    parser.add_argument('--img_resize', type=int, default=224, help='Image resizing (-1 to not resize)')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers for dataloader')

    parser.add_argument('--gene_list', type=str, default='var_50genes.json')
    parser.add_argument('--method', type=str, default='random-forest', help='random-forest or ridge')
    parser.add_argument('--normalize', type=bool, default=True)
    parser.add_argument('--dimreduce', type=str, default=None, help='whenever to perform dimensionality reduction before linear probing, can be "PCA" or None')
    parser.add_argument('--latent_dim', type=int, default=256, help='dimensionality reduction latent dimension')
    parser.add_argument('--encoders', nargs='+', default=["uni_v1_official"], help='All the encoders to benchmark:uni_v1_official,resnet50_trunc,ciga_loader')
    parser.add_argument('--datasets', nargs='+', default=["LUNG"], help='Datasets from source_dataroot to use during benchmark')
    args = parser.parse_args()

    set_random_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    precisions = {'fp16': torch.float16, 
                  'fp32': torch.float32}
    precision = precisions.get(args.precision, torch.float32)

    #### download model checkpoints and dataset ####
    snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=args.weights_root, allow_patterns=['fm_v1/*'])
    snapshot_download(repo_id="MahmoodLab/hest-bench", repo_type='dataset', local_dir=args.source_dataroot, ignore_patterns=['fm_v1/*'])
    hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=os.path.join(args.weights_root, "uni/"))
    hf_hub_download("prov-gigapath/prov-gigapath", filename="pytorch_model.bin", local_dir=os.path.join(args.weights_root, "gigapath/"))

    datasets = args.datasets
    if len(datasets) >= 1 and datasets[0] == '*':
        datasets = os.listdir(get_path(args.source_dataroot))

    #### Setup Save Directory ####
    save_dir = get_path(args.results_dir)
    if args.exp_code is None:
        exp_code = f"run_{get_current_time()}"
    else:
        exp_code = args.exp_code + f"::{get_current_time()}"
    save_dir = os.path.join(save_dir, exp_code)
    os.makedirs(save_dir, exist_ok=True)

    encoders = []
    weights_root = get_path(args.weights_root)
    for enc_name in args.encoders:
        encoders.append(LazyEncoder(enc_name, weights_root=weights_root))

    #### execute fn for each encoders and datasets and dump the results in a nested directory structure ###
    dataset_perfs = []
    for dataset in datasets:
        if dataset not in {"CCRCC", "COAD", "HCC", "IDC", "LUNG", "LYMPH_IDC", "PAAD", "PRAD", "READ", "SKCM"}:
            continue

        source_dataroot = os.path.join(get_path(args.source_dataroot), dataset)
        enc_perfs = []
        for enc in encoders:
            exp_save_dir = os.path.join(save_dir, dataset, enc.name)
            print("exp_save_dir", exp_save_dir)

            os.makedirs(exp_save_dir, exist_ok=True)
            enc_results = predict_folds(args, exp_save_dir, enc, dataset, device, precision, source_dataroot)
            
            enc_perfs.append({
                'encoder_name': enc.name,
                'pearson_mean': enc_results['pearson_mean'], 
                'pearson_std': enc_results['pearson_std'], 
            })
            
        with open(os.path.join(save_dir, dataset, 'enc_results.json'), 'w') as f:
            #enc_perf = sorted(enc_perf.items(), key=lambda x: x[1]['pearson_mean'])
            enc_perfs = sorted(enc_perfs, key=itemgetter('pearson_mean'), reverse=True)
            json.dump({'results': enc_perfs}, f, sort_keys=True, indent=4)
            
        dataset_perfs.append({
            'dataset_name': dataset,
            'results': enc_perfs
        })
        
    perf_per_enc = {}
    for dataset_perf in dataset_perfs:
        for enc_perf in dataset_perf['results']:
            perf_per_enc[enc_perf['encoder_name']] = perf_per_enc.get(enc_perf['encoder_name'], []) + [enc_perf['pearson_mean']]
            
    for key, val in perf_per_enc.items():
        perf_per_enc[key] = np.mean(val)
    perf_per_enc = sorted(perf_per_enc, key=lambda item: item[1], reverse=True)
        
    with open(os.path.join(save_dir, 'dataset_results.json'), 'w') as f:
        json.dump({'results': dataset_perfs, 'average': perf_per_enc}, f, sort_keys=True, indent=4)
