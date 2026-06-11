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

def compute_marginal_jsd(H_base, candidate_histograms, global_histogram, size_corrected=True):
    H_test = H_base.unsqueeze(0) + candidate_histograms
    H_test_sum = torch.sum(H_test, dim=1, keepdim=True)
    P_test = H_test / H_test_sum
    P_global = global_histogram / torch.sum(global_histogram)
    
    M = 0.5 * (P_test + P_global.unsqueeze(0))
    kl_p_m = torch.sum(P_test * torch.log2((P_test + 1e-10) / (M + 1e-10)), dim=1)
    kl_q_m = torch.sum(P_global * torch.log2((P_global + 1e-10) / (M + 1e-10)), dim=1)
    
    if size_corrected:
        return torch.sqrt(0.5 * kl_p_m + 0.5 * kl_q_m) * torch.sqrt(H_test_sum.squeeze(-1))
    else:
        return torch.sqrt(0.5 * kl_p_m + 0.5 * kl_q_m)


def compute_marginal_kld(H_base, candidate_histograms, global_histogram, size_corrected=True):
    H_test = H_base.unsqueeze(0) + candidate_histograms
    H_test_sum = torch.sum(H_test, dim=1, keepdim=True)
    P_test = H_test / H_test_sum
    P_global = global_histogram / torch.sum(global_histogram)
    
    kl_p_q = torch.sum(P_test * torch.log2((P_test + 1e-10) / (P_global.unsqueeze(0) + 1e-10)), dim=1)
    
    if size_corrected:
        return kl_p_q * torch.sqrt(H_test_sum.squeeze(-1))
    else:
        return kl_p_q


def compute_marginal_hellinger(H_base, candidate_histograms, global_histogram, size_corrected=True):
    H_test = H_base.unsqueeze(0) + candidate_histograms
    H_test_sum = torch.sum(H_test, dim=1, keepdim=True)
    P_test = H_test / H_test_sum
    P_global = global_histogram / torch.sum(global_histogram)

    hellinger = torch.sqrt(1.0 - torch.sum(torch.sqrt(P_test * P_global.unsqueeze(0)), dim=1))
    
    if size_corrected:
        return hellinger * torch.sqrt(H_test_sum.squeeze(-1))
    else:
        return hellinger


def compute_marginal_tvd(H_base, candidate_histograms, global_histogram, size_corrected=True):
    H_test = H_base.unsqueeze(0) + candidate_histograms
    H_test_sum = torch.sum(H_test, dim=1, keepdim=True)
    P_test = H_test / H_test_sum
    P_global = global_histogram / torch.sum(global_histogram)

    tvd = 0.5 * torch.sum(torch.abs(P_test - P_global.unsqueeze(0)), dim=1)
    
    if size_corrected:
        return tvd * torch.sqrt(H_test_sum.squeeze(-1))
    else:
        return tvd