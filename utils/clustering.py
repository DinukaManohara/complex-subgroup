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
import pykeops
from pykeops.torch import LazyTensor

from datasets.metadata import uci_datasets_info


def keops_knn_geo(dataset: torch.Tensor, k: int = 10, device: str = 'cuda'):
    N, D = dataset.shape

    x_i = LazyTensor(dataset.view(N, 1, D))
    y_j = LazyTensor(dataset.view(1, N, D))

    D_ij = (x_i - y_j).abs().sum(dim=-1)
    idx = D_ij.argKmin(K=k, dim=1)

    return idx

def keops_knn_pal_foe(histogram: torch.Tensor, position: torch.Tensor, k: int = 10, device: str = 'cuda'):
    N, hD = histogram.shape
    N, pD = position.shape
    
    density = histogram / histogram.sum(dim=1, keepdim=True)

    xp_i = LazyTensor(position.view(N, 1, pD))
    yp_j = LazyTensor(position.view(1, N, pD))

    MAN = (xp_i - yp_j).abs().sum(dim=-1) / pD

    P = LazyTensor(density.view(N, 1, hD))
    Q = LazyTensor(density.view(1, N, hD))

    M = 0.5 * (P + Q)
    
    eps = 1e-12
    JSD = (0.5 * ( (P * (P + eps).log() - P * (M + eps).log()).sum(dim=-1) + (Q * (Q + eps).log() - Q * (M + eps).log()).sum(dim=-1) )).sqrt()

    pal_D_ij = (1.0 - MAN) + JSD
    foe_D_ij = MAN + (torch.sqrt(torch.log(torch.tensor(2, device=device))) - JSD)

    pal_idx = pal_D_ij.argKmin(K=k, dim=1)
    foe_idx = foe_D_ij.argKmin(K=k, dim=1)

    return pal_idx, foe_idx

def keops_knn_pal(histogram: torch.Tensor, position: torch.Tensor, k: int = 10, device: str = 'cuda'):
    N, hD = histogram.shape
    N, pD = position.shape
    
    density = histogram / histogram.sum(dim=1, keepdim=True)

    xp_i = LazyTensor(position.view(N, 1, pD))
    yp_j = LazyTensor(position.view(1, N, pD))

    MAN = (xp_i - yp_j).abs().sum(dim=-1) / pD

    P = LazyTensor(density.view(N, 1, hD))
    Q = LazyTensor(density.view(1, N, hD))

    M = 0.5 * (P + Q)
    
    eps = 1e-12
    JSD = (0.5 * ( (P * (P + eps).log() - P * (M + eps).log()).sum(dim=-1) + (Q * (Q + eps).log() - Q * (M + eps).log()).sum(dim=-1) )).sqrt()

    pal_D_ij = (1.0 - MAN) + JSD

    pal_idx = pal_D_ij.argKmin(K=k, dim=1)

    return pal_idx


def KMeans(x, K=10, Niter=10, tol=1e-6, use_cuda=True, verbose=True):
    """Implements Lloyd's algorithm for the Euclidean metric."""

    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space

    # c = x[:K, :].clone()  # Simplistic initialization for the centroids
    c_indices = torch.randperm(len(x))[:K] # Get K random unique indices
    c = x[c_indices].clone()

    x_i = LazyTensor(x.view(N, 1, D))  # (N, 1, D) samples
    c_j = LazyTensor(c.view(1, K, D))  # (1, K, D) centroids

    # K-means loop:
    # - x  is the (N, D) point cloud,
    # - cl is the (N,) vector of class labels
    # - c  is the (K, D) cloud of cluster centroids
    for i in range(Niter):
        c_old = c.clone()

        # E step: assign points to the closest cluster -------------------------
        D_ij = ((x_i - c_j) ** 2).sum(-1)  # (N, K) symbolic squared distances
        cl = D_ij.argmin(dim=1).long().view(-1)  # Points -> Nearest cluster

        # M step: update the centroids to the normalized cluster average: ------
        # Compute the sum of points per cluster:
        c.zero_()
        c.scatter_add_(0, cl[:, None].repeat(1, D), x)

        # Divide by the number of points per cluster:
        Ncl = torch.bincount(cl, minlength=K).type_as(c).view(K, 1)
        c /= Ncl  # in-place division to compute the average

        movement = (c - c_old).norm()
        if movement < tol:
            print(f"Converged at iteration {i}")
            break

    if verbose:  # Fancy display -----------------------------------------------
        if use_cuda:
            torch.cuda.synchronize()
        end = time.time()
        print(
            f"K-means for the Manhattan metric with {N:,} points in dimension {D:,}, K = {K:,}:"
        )
        print(
            "Timing for {} iterations: {:.5f}s = {} x {:.5f}s\n".format(
                Niter, end - start, Niter, (end - start) / Niter
            )
        )

    return cl, c, end - start

