import sys
import os
import gc
import pandas as pd
import random
import copy
import pickle
import math
import numpy as np
import time
import torch
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from datasets.metadata import uci_datasets_info
from utils.process_data import preprocess_dataframe
from utils.clustering import *
from Walker import ActiveDPGMMWalker
from utils.measures import *

torch.manual_seed(20260421)
rng = np.random.default_rng(seed=20260421)

num_bins = 32

# pocket_sizes_dict = {
#     16000: [8, 16, 32, 64],
#     32000: [16, 32, 64, 128],
#     64000: [32, 64, 128, 256],
#     128000: [64, 128, 256, 512],
#     256000: [128, 256, 512, 1024],
#     512000: [256, 512, 1024, 2048],
#     1024000: [512, 1024, 2048, 4096],
#     2048000: [1024, 2048, 4096, 8192],
#     4096000: [2048, 4096, 8192, 16384]
# }

pocket_sizes_dict = {
    16000: [1024, 2048, 4096, 512],
    32000: [1024, 2048, 4096, 512],
    64000: [1024, 2048, 4096, 512],
    128000: [1024, 2048, 4096, 512],
    256000: [1024, 2048, 4096, 512],
    512000: [1024, 2048, 4096, 512],
    1024000: [1024, 2048, 4096, 512]
}

# pocket_sizes_dict = {
#     16000: [4096, 2048, 1024, 512],
#     32000: [4096, 2048, 1024, 512],
#     64000: [4096, 2048, 1024, 512],
#     128000: [4096, 2048, 1024, 512],
#     256000: [4096, 2048, 1024, 512],
#     512000: [4096, 2048, 1024, 512],
#     1024000: [4096, 2048, 1024, 512]
# }

# pocket_sizes_dict = {
#     16000: [4],
#     32000: [8],
#     64000: [16],
#     128000: [32],
#     256000: [64],
#     512000: [128],
#     1024000: [256]
# }

measures = {
    'jsd':compute_marginal_jsd,
    'kld':compute_marginal_kld,
    'hellinger':compute_marginal_hellinger,
    'tvd':compute_marginal_tvd,
}

if len(sys.argv) == 2+1:
    name = str(sys.argv[1])
    size = int(sys.argv[2])
else:
    print("Expected arguments are not given.")
    exit()

beta = 3.0

if name == 'C':
    X, _ = make_blobs(n_samples=16000, centers=2000, n_features=size, cluster_std=1.5, random_state=20260421)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    tar_x = rng.random((16000, 1))
    Z = np.concatenate([tar_x, X_scaled], axis=1)

    df = pd.DataFrame(Z, columns=[f'x{i}' for i in range(size+1)])
    condition = ((df['x1'] > 0.9) & (df['x2'] < 0.3)) | ((df['x1'] < 0.2) & (df['x2'] > 0.9)) | ((0.4 < df['x1']) & (df['x1'] < 0.6) & (0.2 < df['x2']) & (df['x2'] < 0.4))
    filtered_df = df[condition]
    filtered_indices = filtered_df.index
    df.loc[filtered_indices, 'x0'] = np.random.normal(loc=1.5, scale=0.5, size=len(filtered_indices))

    tensor_data = torch.tensor(df.values, dtype=torch.float32, device='cuda')

    tensor_data = (tensor_data - tensor_data.min(dim=0).values) / tensor_data.max(dim=0).values
    col_widths = (tensor_data.max(dim=0).values + 1e-6 - tensor_data.min(dim=0).values) / num_bins

    pos = tensor_data[:, 1:].contiguous()
    tar = tensor_data[:, 0].unsqueeze(1).contiguous()

    hist_idx = (tar // col_widths[0]).long().squeeze(-1)
    hists = torch.zeros((16000, num_bins), device='cuda')
    hists[torch.arange(16000), hist_idx] = 1.0

    pocket_sizes = [8, 16, 32, 64]
    beta = 12.0
elif name == 'N':
    X, _ = make_blobs(n_samples=size, centers=2000, n_features=2, cluster_std=1.5, random_state=20260421)
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)
    tar_x = rng.random((size, 1))
    Z = np.concatenate([tar_x, X_scaled], axis=1)
    df = pd.DataFrame(Z, columns=[f'x{i}' for i in range(3)])
    condition = ((df['x1'] > 0.9) & (df['x2'] < 0.3)) | ((df['x1'] < 0.2) & (df['x2'] > 0.9)) | ((0.4 < df['x1']) & (df['x1'] < 0.6) & (0.2 < df['x2']) & (df['x2'] < 0.4))
    filtered_df = df[condition]
    filtered_indices = filtered_df.index
    df.loc[filtered_indices, 'x0'] = np.random.normal(loc=1.5, scale=0.5, size=len(filtered_indices))

    tensor_data = torch.tensor(df.values, dtype=torch.float32, device='cuda')

    tensor_data = (tensor_data - tensor_data.min(dim=0).values) / tensor_data.max(dim=0).values
    col_widths = (tensor_data.max(dim=0).values + 1e-6 - tensor_data.min(dim=0).values) / num_bins

    pos = tensor_data[:, 1:].contiguous()
    tar = tensor_data[:, 0].unsqueeze(1).contiguous()

    hist_idx = (tar // col_widths[0]).long().squeeze(-1)
    hists = torch.zeros((size, num_bins), device='cuda')
    hists[torch.arange(size), hist_idx] = 1.0

    pocket_sizes = pocket_sizes_dict[size]
    beta = 6.0

summary = [['pocket_size', 'runtime']]
for pocket_size in pocket_sizes:
    pocket_details = {}
    with open(f'./perceptive_graphs/synthetic_{name}{size}_p{pocket_size}.pkl', 'rb') as file:
        pocket_details = pickle.load(file)

    pockets = pocket_details['pockets']
    c = pocket_details['midpoints']
    cl_hists = pocket_details['pocket_hists']

    neighborhood = math.ceil(math.sqrt(len(pockets)) / 2)

    geo_index = keops_knn_geo(c, k=neighborhood + 1)
    pal_index = keops_knn_pal(cl_hists, c, k=neighborhood)
    knn_index = torch.cat([geo_index[:, 1:], pal_index], dim=-1)
    neighbor_types = torch.tensor([0 if i < neighborhood else 1 for i in range(knn_index.shape[1])], dtype=bool, device='cuda').unsqueeze(0).repeat(knn_index.shape[0], 1)

    dataset_mean = pos.mean(dim=0)
    base_dataset_cov = torch.cov(pos.T)

    target_items = 0.1 * pos.shape[0]

    default_base = torch.zeros(num_bins, device='cuda')

    measure = 'jsd'

    compute_marginal_stat = measures[measure]

    untouched_ids = set(range(len(pockets)))

    print(f'\ndataset={name} {name}={size}')
    print(f'========================================================================================================')

    print('WARM-UP')
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

        alpha = 1.0
        lambda_reg = 1.0

        walker = ActiveDPGMMWalker(
            num_pockets=len(pockets), 
            original_datapoints=pos, 
            hists=hists,
            pocket_hists=cl_hists, 
            pockets_list=pockets, 
            initial_mu=dataset_mean, 
            initial_sigma=base_dataset_cov * beta,
            alpha=alpha, 
            lambda_reg=lambda_reg
        )

        init_node = random.choice(list(untouched_ids))

        try:
            best_nodes, clusters, sub_hist = walker.walk(
                pocket_midpoints=c,
                all_midpoints=pos,
                adj_knn=knn_index, 
                adj_types=neighbor_types,
                start_pocket_idx=init_node,
                compute_marginal_stat=compute_marginal_stat,
                target_items=target_items
            )

        except Exception as e:
            print(e)
    print('WARM-UP DONE.')
    
    for _ in range(3):
        gc.collect()
        torch.cuda.empty_cache()

        alpha = 1.0
        lambda_reg = 1.0

        walker = ActiveDPGMMWalker(
            num_pockets=len(pockets), 
            original_datapoints=pos, 
            hists=hists,
            pocket_hists=cl_hists, 
            pockets_list=pockets, 
            initial_mu=dataset_mean, 
            initial_sigma=base_dataset_cov * beta,
            alpha=alpha, 
            lambda_reg=lambda_reg
        )

        init_node = random.choice(list(untouched_ids))

        try:
            start = time.time()

            best_nodes, clusters, sub_hist = walker.walk(
                pocket_midpoints=c,
                all_midpoints=pos,
                adj_knn=knn_index, 
                adj_types=neighbor_types,
                start_pocket_idx=init_node,
                compute_marginal_stat=compute_marginal_stat,
                target_items=target_items
            )

            end = time.time()
            duration = end - start

            # untouched_ids = untouched_ids - set(best_nodes)

            summary.append([pocket_size, duration])

        except Exception as e:
            print(e)

with open(f'./results/mod/synthetic_{name}{size}.pkl', 'wb') as f:
    pickle.dump(summary, f)

# with open(f'./results/mod/synthetic_{name}{size}_v4000.pkl', 'wb') as f:
#     pickle.dump(summary, f)
