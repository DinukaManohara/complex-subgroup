import os
import glob
from datasets.metadata import uci_datasets_info
import pickle
from functools import reduce
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from scipy.stats import chi2
from datasets.metadata import uci_datasets_info
from utils.process_data import preprocess_dataframe
from utils.measures import *
from utils.clustering import *


def get_confidence_interval_mask(all_midpoints, clusters, m_thresholds):
    N, D = all_midpoints.size()
    device = all_midpoints.device
    K = len(clusters)
    
    # Handle the edge case where no clusters exist yet
    if K == 0:
        return torch.zeros(N, dtype=torch.bool, device=device)
        
    mu_batch = torch.stack([c['E_mu'] for c in clusters])       # Shape: (K, D)
    Sigma_batch = torch.stack([c['E_Sigma'] for c in clusters]) # Shape: (K, D, D)
    thresh_batch = torch.tensor(m_thresholds, device=device)    # Shape: (K,)
    
    centered_data = all_midpoints.unsqueeze(0) - mu_batch.unsqueeze(1)
    
    jitter = torch.eye(D, device=device) * 1e-6
    L_batch = torch.linalg.cholesky(Sigma_batch + jitter)       # Shape: (K, D, D)

    L_batch = torch.linalg.cholesky(Sigma_batch)
    
    y_batch = torch.linalg.solve_triangular(L_batch, centered_data.transpose(1, 2), upper=False)
    
    mahal_sq_batch = torch.sum(y_batch ** 2, dim=1)
    
    thresh_sq_batch = (thresh_batch ** 2).unsqueeze(1)
    
    # Evaluate boolean mask for all points against all clusters simultaneously: (K, N)
    cluster_masks = mahal_sq_batch <= thresh_sq_batch
    
    # Perform a logical OR down the K dimension (dim=0) to see if a point is covered by ANY cluster
    # Resulting global_coverage_mask: (N,)
    global_coverage_mask = torch.any(cluster_masks, dim=0)
    
    return global_coverage_mask

def concretise(pos, clusters, hists, compute_marginal_stat, default_base, global_hist, support_thres=0.1):
    prev_support = 1.0 #row['support']
    upper = 1.0 #row['confidence']
    lower = 0.0
    thres = 0
    score = 0
    sub_hist = None
    coverage_nodes = None

    while True:
        percentage = (lower + upper) / 2
        # print(prev_support, percentage)
        try:
            thres = np.sqrt(chi2.ppf(percentage, df=pos.shape[1]))
            coverage_mask = get_confidence_interval_mask(pos, clusters, m_thresholds=[thres]*len(clusters))
            coverage_nodes = torch.nonzero(coverage_mask).squeeze(-1)
            sub_hist = hists[coverage_nodes].sum(dim=0)

            score = compute_marginal_stat(default_base, sub_hist.unsqueeze(0), global_hist, size_corrected=False)[0].item()

            support = sub_hist.sum().item() / pos.shape[0]

            if support > support_thres:
                lower = lower
                upper = percentage
            elif support == support_thres:
                break
            else:
                lower = percentage
                upper = upper

            # print(support)

            # if prev_support == support:
            #     if support >= support_thres:
            #         break
            #     else:
            #         prev_support = support
            # else:
            #     prev_support = support

            

            if upper - lower < 1e-12:
                break
        except Exception as e:
            print(e)
            raise e

    return score, sub_hist, thres