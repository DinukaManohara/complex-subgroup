import sys
import os
import pandas as pd
import random
import copy
import pickle
import math
import numpy as np
import time
import torch
from datasets.metadata import uci_datasets_info
from utils.process_data import preprocess_dataframe
from utils.clustering import *
from Walker import ActiveDPGMMWalker
from utils.measures import *
from utils.concretise import concretise

name = None
pocket_size = None
neighborhood = None

if len(sys.argv) == 1+1:
    pocket_size = int(sys.argv[1])
else:
    print("Expected arguments are not given.")
    exit()

measure = 'jsd'

measures = {
    'jsd':compute_marginal_jsd,
    'kld':compute_marginal_kld,
    'hellinger':compute_marginal_hellinger,
    'tvd':compute_marginal_tvd,
}

with open("./datasets/case_study/train_df_oof.pkl", "rb") as f:
    df = pickle.load(f)[['x1', 'x2', 'x3', 'logloss']]
    # df = pickle.load(f)[['x1', 'x3', 'x4', 'logloss']]
    # df = pickle.load(f)[['x6', 'x7', 'x8', 'logloss']]

df, mapping, _ = preprocess_dataframe(df, 'logloss', [], [])
tensor_data = torch.tensor(df.to_numpy(), dtype=torch.float32, device='cuda')

if df['x0'].unique().shape[0] <= 2:
        col_width = 0.5
else:
    STD = df['x0'].std()
    col_width = 3.5 * (STD / math.cbrt(tensor_data.shape[0]))  # Applying the Scott's rule.

num_bins = math.ceil((df['x0'].max() - df['x0'].min()) / col_width)

pos = tensor_data[:, 1:].contiguous()
tar = tensor_data[:, 0].unsqueeze(1).contiguous()

hist_idx = torch.clamp((tar // col_width).long().squeeze(-1), max=num_bins - 1)
hists = torch.zeros((df.shape[0], num_bins), device='cuda')
hists[torch.arange(df.shape[0]), hist_idx] = 1.0

pocket_details = {}
with open(f'./perceptive_graphs/case_study_{pocket_size}.pkl', 'rb') as file:
    pocket_details = pickle.load(file)

pockets = pocket_details['pockets']
c = pocket_details['midpoints']
cl_hists = pocket_details['pocket_hists']

neighborhood = math.ceil(math.sqrt(len(pockets)) / 2)
# neighborhood = 2 * math.ceil(math.sqrt(len(pockets)) / 2) # Enable this to run without Pal neighbors

geo_index = keops_knn_geo(c, k=neighborhood + 1)
pal_index = keops_knn_pal(cl_hists, c, k=neighborhood)
knn_index = torch.cat([geo_index[:, 1:], pal_index], dim=-1)
neighbor_types = torch.tensor([0 if i < neighborhood else 1 for i in range(knn_index.shape[1])], dtype=bool, device='cuda').unsqueeze(0).repeat(knn_index.shape[0], 1)

dataset_mean = pos.mean(dim=0)
base_dataset_cov = torch.cov(pos.T)

support_thres = 0.20

target_items = support_thres * pos.shape[0]

default_base = torch.zeros(num_bins, device='cuda')

# for measure in measures:
# if os.path.isfile(f'./results/mod/{name}_p{pocket_size}_{measure}.pkl'):
#     print(f'{name}_p{pocket_size}_{measure} exists. Skipping...')
#     continue

compute_marginal_stat = measures[measure]

summary = [['alpha', 'beta', 'lambda', 'score', 'num_nodes', 'num_clusters', 'duration']]

best_scores = [-float('inf'), -float('inf'), -float('inf')]
best_details = [None, None, None]
best_configs = [None, None, None]

for seed in [20260501, 20260502, 20260503]:
    torch.manual_seed(seed)
    
    for alpha in [1.0]:
        for beta in [0.9, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0]:
            for lambda_reg in [0.0, 0.25, 0.5, 0.75, 1.0]:
                untouched_ids = set(range(len(pockets)))
                
                print(f'\npocket_size={pocket_size} alpha={alpha} beta={beta} lambda={lambda_reg} measure={measure}')
                print(f'========================================================================================================')
                
                # init_scores = None
                # untouched_ids = None
                for turn in range(3):
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
                    
                    # if turn == 0:
                    #     init_scores = compute_marginal_stat(default_base, walker.pocket_hists, walker.global_histogram)
                    #     the_max = init_scores.max()
                    #     candidate_pocket_ids = (torch.abs(init_scores - the_max) <= 1e-6).nonzero()
                    #     untouched_ids = set(candidate_pocket_ids.squeeze(-1).tolist())
                    #     print(f"Found {len(untouched_ids)} candidate init nodes.")
                    
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

                        untouched_ids = untouched_ids - set(best_nodes)

                        try:
                            # score = compute_marginal_stat(default_base, sub_hist.unsqueeze(0), walker.global_histogram, size_corrected=False)[0].item()
                            score, concrete_sub_hist, thres = concretise(pos, clusters, hists, compute_marginal_stat, default_base, walker.global_histogram, support_thres=support_thres)
                            print(score)
                        except Exception as e:
                            continue

                        summary.append([alpha, beta, lambda_reg, score, len(best_nodes), len(clusters), duration])

                        if score > best_scores[-1]:
                            best_scores[-1] = score
                            best_details[-1] = {'best_nodes': best_nodes, 'clusters': clusters, 'sub_hist': concrete_sub_hist, 'threshold': thres}
                            best_configs[-1] = (init_node, alpha, beta, lambda_reg)
                            idx = sorted(range(len(best_scores)), key=best_scores.__getitem__, reverse=True)
                            best_scores = [best_scores[j] for j in idx]
                            best_details = [best_details[j] for j in idx]
                            best_configs = [best_configs[j] for j in idx]
                        elif abs(score - best_scores[-1]) <= 1e-3:
                            if len(best_details[-1]['clusters']) > len(clusters):
                                best_scores[-1] = score
                                best_details[-1] = {'best_nodes': best_nodes, 'clusters': clusters, 'sub_hist': concrete_sub_hist, 'threshold': thres}
                                best_configs[-1] = (init_node, alpha, beta, lambda_reg)
                                idx = sorted(range(len(best_scores)), key=best_scores.__getitem__, reverse=True)
                                best_scores = [best_scores[j] for j in idx]
                                best_details = [best_details[j] for j in idx]
                                best_configs = [best_configs[j] for j in idx]

                    except Exception as e:
                        print(e)

with open(f'./results/mod/case_study_p{pocket_size}.pkl', 'wb') as f:
    pickle.dump(
        {
            'best_scores': best_scores,
            'best_details': best_details,
            'best_configs': best_configs,
            'summary': summary
        }, f)
