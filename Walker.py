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
import torch.distributions as dist

class ActiveDPGMMWalker:
    def __init__(self, num_pockets, original_datapoints, hists, pocket_hists, pockets_list, initial_mu, initial_sigma, alpha, lambda_reg, device='cuda'):
        self.num_pockets = num_pockets
        self.D = original_datapoints.size(1)
        self.device = device
        
        self.original_datapoints = original_datapoints
        self.pockets_list = pockets_list

        self.pocket_hists = pocket_hists.to(self.device)
        self.all_hists = hists.to(device)
        self.global_histogram = self.pocket_hists.sum(dim=0)
        
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        
        # --- DYNAMIC PADDING TENSOR ---
        # Converts variable-length pocket lists into a dense GPU tensor for vectorized item evaluation
        max_pocket_size = max(p.size(0) for p in pockets_list if p.numel() > 0)
        self.pockets_tensor = torch.full((self.num_pockets, max_pocket_size), -1, dtype=torch.long, device=self.device)
        for i, p in enumerate(pockets_list):
            if p.numel() > 0:
                self.pockets_tensor[i, :p.size(0)] = p.to(self.device)
        
        # --- EXPLICIT NIW PRIOR INITIALIZATION ---
        self.mu_0 = initial_mu.to(self.device)
        self.kappa_0 = 0.5
        self.nu_0 = self.D + 2.0 
        
        jitter = torch.eye(self.D, device=self.device) * 1e-6
        safe_sigma = initial_sigma.to(self.device) + jitter
        self.Psi_0 = safe_sigma * (self.nu_0 - self.D - 1.0)
        
        self.expected_mu_0 = self.mu_0
        self.expected_Sigma_0 = safe_sigma
        
        # Graph State Masks 
        self.in_subgroup_pockets = torch.zeros(self.num_pockets, dtype=torch.bool, device=self.device)
        self.in_frontier_pockets = torch.zeros(self.num_pockets, dtype=torch.bool, device=self.device)
        self.is_frontier_pal = torch.zeros(self.num_pockets, dtype=torch.bool, device=self.device)
        
        self.clusters = []

    def init_cluster(self, pocket_datapoints, datapoint_ids):
        """Initializes a new cluster using the Dynamic Local Prior (centered on the pocket)."""
        l = pocket_datapoints.size(0)
        y_bar = pocket_datapoints.mean(dim=0)
        
        centered = pocket_datapoints - y_bar
        S = torch.mm(centered.t(), centered)
        
        kappa_new = self.kappa_0 + l
        nu_new = self.nu_0 + l
        
        # Dynamic Local Prior: Center is strictly the pocket midpoint, outer product penalty is 0
        mu_new = y_bar
        Psi_new = self.Psi_0 + S 

        jitter = torch.eye(self.D, device=self.device) * 1e-6
        
        cluster_state = {
            'N': l, 'kappa': kappa_new, 'nu': nu_new, 'mu': mu_new, 'Psi': Psi_new,
            'E_mu': mu_new, 'E_Sigma': (Psi_new / (nu_new - self.D - 1.0)) + jitter, 'items': datapoint_ids.tolist()
        }
        self.clusters.append(cluster_state)

    def update_cluster(self, cluster_idx, pocket_datapoints, datapoint_ids):
        """Standard NIW Batch Update using raw items."""
        c = self.clusters[cluster_idx]
        
        l = pocket_datapoints.size(0)
        y_bar = pocket_datapoints.mean(dim=0)
        
        centered = pocket_datapoints - y_bar
        S = torch.mm(centered.t(), centered)
        
        kappa_new = c['kappa'] + l
        nu_new = c['nu'] + l
        
        diff = (y_bar - c['mu']).unsqueeze(1)
        Psi_new = c['Psi'] + S + ((c['kappa'] * l) / kappa_new) * torch.mm(diff, diff.t())
        mu_new = (c['kappa'] * c['mu'] + l * y_bar) / kappa_new
        
        c['N'] += l
        c['kappa'] = kappa_new
        c['nu'] = nu_new
        c['mu'] = mu_new
        c['Psi'] = Psi_new
        
        c['E_mu'] = mu_new
        jitter = torch.eye(self.D, device=self.device) * 1e-6
        c['E_Sigma'] = (Psi_new / (nu_new - self.D - 1.0)) + jitter

        c['items'].extend(datapoint_ids.tolist())

    def calculate_soft_H_base(self, all_midpoints):
        if not self.clusters:
            return torch.zeros(self.pocket_hists.size(1), device=self.device)
            
        # global_weights = torch.zeros(self.num_pockets, device=self.device)
        global_weights = torch.zeros(all_midpoints.shape[0], device=self.device)
        
        for c in self.clusters:
            mvn = dist.MultivariateNormal(c['E_mu'], c['E_Sigma'])
            
            log_probs = mvn.log_prob(all_midpoints)
            peak_log_prob = mvn.log_prob(c['E_mu'])
            
            # Normalization equates strictly to exp(-0.5 * Mahalanobis^2)
            soft_membership = torch.exp(log_probs - peak_log_prob)
            
            # Enforce Disjoint Rules: Fuzzy OR (Max)
            global_weights = torch.max(global_weights, soft_membership)
            
        # Multiply (num_pockets, C) by (num_pockets, 1) and sum to (C,)
        soft_H_base = torch.sum(self.all_hists * global_weights.unsqueeze(-1), dim=0)
        return soft_H_base

    def walk(self, pocket_midpoints, all_midpoints, adj_knn, adj_types, target_items, compute_marginal_stat, start_pocket_idx=None):
        if start_pocket_idx is None:
            start_pocket_idx = torch.randint(0, self.num_pockets, (1,)).item()
            print(start_pocket_idx)
            
        self.in_subgroup_pockets[start_pocket_idx] = True
        
        initial_dp_ids = self.pockets_list[start_pocket_idx]
        initial_datapoints = self.original_datapoints[initial_dp_ids]
        
        self.init_cluster(initial_datapoints, initial_dp_ids)
        
        neighbors = adj_knn[start_pocket_idx]
        neighbor_mask = (neighbors != -1) & (~self.in_subgroup_pockets[neighbors])
        valid_neighbors = neighbors[neighbor_mask]
        self.in_frontier_pockets[valid_neighbors] = True
        # print(valid_neighbors)
        valid_naighbors_type = adj_types[start_pocket_idx][neighbor_mask]
        # print(valid_naighbors_type)
        self.is_frontier_pal[valid_neighbors] = valid_naighbors_type
        
        iteration = 0
        total_datapoints_covered = initial_datapoints.size(0)

        current_soft_H_base = self.pocket_hists[start_pocket_idx]
        # current_soft_H_base = self.calculate_soft_H_base(all_midpoints)

        print('Starting...')

        while total_datapoints_covered < target_items:
        # while current_soft_H_base.sum() < target_items:
            if len(self.clusters) >= 100:
                print('Too many components.')
                break
            
            frontier_tensor = self.in_frontier_pockets.nonzero(as_tuple=True)[0]
            
            if frontier_tensor.numel() == 0:
                print("Frontier completely exhausted.")
                break
                
            # --- 1. Dynamic Continuous Footprint ---
            # current_soft_H_base = self.calculate_soft_H_base(all_midpoints)
            # print()

            # current_soft_H_base = self.pocket_hists[self.in_subgroup_pockets.nonzero().squeeze(-1)].sum(dim=0)
            

            # --- 2. Lightning Fast JSD ---
            H_pocket_total = self.pocket_hists[frontier_tensor] # (F, C)
            stat_scores = compute_marginal_stat(current_soft_H_base, H_pocket_total, self.global_histogram)

            min_stat_val = torch.min(stat_scores)
            max_stat_val = torch.max(stat_scores)
            stat_scores = (stat_scores - min_stat_val) / (max_stat_val - min_stat_val + 1e-12)

            # --- 3. Memory-Optimized Spatial Regularizer (STRICTLY EXISTING CLUSTERS) ---
            F_datapoints = self.pockets_tensor[frontier_tensor] 
            valid_dp_mask = F_datapoints != -1
            
            safe_F_datapoints = F_datapoints.clamp(min=0) 
            pocket_items = self.original_datapoints[safe_F_datapoints] # (F, max_l, D)
            
            cluster_log_weighted_joints = []
            
            # print(f'Iter through {len(self.clusters)} clusters. Pocket size={pocket_items.shape}')
            # print('==============================================')
            for c in self.clusters:
                # print('Creating distributions.')
                mvn = dist.MultivariateNormal(c['E_mu'], c['E_Sigma'])
                # print('Calculating Log-Probs.')
                ll = mvn.log_prob(pocket_items) 
                
                # Mask out padding slots
                # print('Masking.')
                masked_ll = torch.where(valid_dp_mask, ll, torch.zeros_like(ll))
                joint_ll = masked_ll.sum(dim=1) # (F,)
                
                # print('Weights Calculation.')
                w = torch.tensor(c['N'] / (total_datapoints_covered + self.alpha), device=self.device)
                cluster_log_weighted_joints.append(joint_ll + torch.log(w + 1e-15))
            # print('Done.')
            # print('==============================================')


                
            log_weighted_likelihoods_K = torch.stack(cluster_log_weighted_joints, dim=1)
            log_dpgmm_scores = torch.logsumexp(log_weighted_likelihoods_K, dim=1) # (F,)

            # --- 4. Acquisition & Harmonization ---
            # Min-Max Scaling to prevent Spatial scores (-500) from obliterating JSD (0.1)
            min_val = torch.min(log_dpgmm_scores)
            max_val = torch.max(log_dpgmm_scores)
            spatial_scores_scaled = (log_dpgmm_scores - min_val) / (max_val - min_val + 1e-10)

            acquisition_scores = stat_scores + self.lambda_reg * spatial_scores_scaled
            best_idx_in_frontier = torch.argmax(acquisition_scores).item()
            
            best_pocket_idx = frontier_tensor[best_idx_in_frontier].item()
            best_midpoint = pocket_midpoints[best_pocket_idx]
            
            best_datapoint_ids = self.pockets_list[best_pocket_idx]
            best_datapoints = self.original_datapoints[best_datapoint_ids]

            # --- 5. Hard Assignment Evaluation (Including Dynamic Local Prior) ---
            winner_K_log_probs = log_weighted_likelihoods_K[best_idx_in_frontier] # (K,)
            
            # The Dynamic Prior is centered exactly on the candidate pocket's midpoint
            mvn_prior = dist.MultivariateNormal(best_midpoint, self.expected_Sigma_0)
            w_prior = torch.tensor(self.alpha / (total_datapoints_covered + self.alpha), device=self.device)
            
            # Evaluate prior strictly for the winning unpadded datapoints
            prior_ll = mvn_prior.log_prob(best_datapoints).sum() + torch.log(w_prior + 1e-15)
            
            log_component_probs = torch.cat([winner_K_log_probs, prior_ll.unsqueeze(0)])
            candidate_cluster = torch.argmax(log_component_probs).item()


            # --- 6. Macro-Graph Updates ---
            added_volume = best_datapoints.size(0)
            total_datapoints_covered += added_volume
            
            self.in_frontier_pockets[best_pocket_idx] = False
            self.in_subgroup_pockets[best_pocket_idx] = True
            
            new_neighbors = adj_knn[best_pocket_idx]
            new_neighbor_mask = (new_neighbors != -1) & (~self.in_subgroup_pockets[new_neighbors])
            valid_new_neighbors = new_neighbors[new_neighbor_mask]
            valid_new_naighbors_type = adj_types[best_pocket_idx][new_neighbor_mask]
            self.in_frontier_pockets[valid_new_neighbors] = True
            self.is_frontier_pal[valid_new_neighbors] = valid_new_naighbors_type

            # --- 7. DPGMM Parameter Update ---
            action = 'Existing'
            if candidate_cluster == len(self.clusters):
                if self.is_frontier_pal[best_pocket_idx]:
                    self.init_cluster(best_datapoints, best_datapoint_ids)
                    action = 'New'
                else:
                    assigned_cluster = torch.argmax(winner_K_log_probs).item()
                    self.update_cluster(assigned_cluster, best_datapoints, best_datapoint_ids)
            else:
                # assigned_cluster = torch.argmax(winner_K_log_probs).item()
                self.update_cluster(candidate_cluster, best_datapoints, best_datapoint_ids)

            
            current_soft_H_base = self.pocket_hists[self.in_subgroup_pockets.nonzero().squeeze(-1)].sum(dim=0)

            # current_soft_H_base = self.calculate_soft_H_base(all_midpoints)

            # if iteration % 10 == 0:
            #     print(f"Datapoints: {total_datapoints_covered} | Action: {action} | Clusters: {len(self.clusters)}")

            iteration += 1

        subgroup_pockets = self.in_subgroup_pockets.nonzero(as_tuple=True)[0].tolist()
        print(f"Search complete. Iterations: {iteration}. Clusters: {len(self.clusters)}. Pockets Acquired: {len(subgroup_pockets)}.")
              
        return subgroup_pockets, self.clusters, current_soft_H_base




