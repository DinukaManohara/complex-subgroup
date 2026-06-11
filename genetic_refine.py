import os
import glob
import time
import pickle
import pandas as pd
import numpy as np
from scipy.stats import chi2
import torch
from datasets.metadata import uci_datasets_info
from utils.process_data import preprocess_dataframe
from utils.measures import *
import pygad

measure = 'jsd'
num_bins = 64
default_base = torch.zeros(num_bins, device='cuda')
compute_marginal_stat = compute_marginal_jsd

def get_confidence_interval_mask(all_midpoints, clusters, m_thresholds):
    """
    Fully vectorized computation of Mahalanobis coverage masks.
    """
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
    
    y_batch = torch.linalg.solve_triangular(L_batch, centered_data.transpose(1, 2), upper=False)
    
    mahal_sq_batch = torch.sum(y_batch ** 2, dim=1)
    
    thresh_sq_batch = (thresh_batch ** 2).unsqueeze(1)
    
    # Evaluate boolean mask for all points against all clusters simultaneously: (K, N)
    cluster_masks = mahal_sq_batch <= thresh_sq_batch
    
    # Perform a logical OR down the K dimension (dim=0) to see if a point is covered by ANY cluster
    # Resulting global_coverage_mask: (N,)
    global_coverage_mask = torch.any(cluster_masks, dim=0)
    
    return global_coverage_mask

def on_generation(ga_instance):
    if ga_instance.generations_completed % 10 == 0:
        print(f"Generation = {ga_instance.generations_completed} | Fitness = {ga_instance.best_solution(pop_fitness=ga_instance.last_generation_fitness)[1]}")

# for name in uci_datasets_info.keys():
for name in ['raisin']:
    # if os.path.isfile(f'./results/genetic/{name}_genetic_{measure}.pkl'):
    #     print(f'{name}_genetic_{measure}.pkl exists. Skipping...')
    #     continue

    file = f'./results/mod/{name}_{measure}_overall_best.pkl'
    if os.path.isfile(file):
        print('\n==================================================================================================')
        print(f'RUNNING FOR DATASET {name}')
        print('==================================================================================================')
        with open(file, 'rb') as f:
            results = pickle.load(f)
    else:
        continue
    
    df = pd.read_csv(f'./datasets/{name}.csv', delimiter=uci_datasets_info[name]['delim'])
    df, mapping, _ = preprocess_dataframe(df, uci_datasets_info[name]['target_variable'][0], uci_datasets_info[name]['categorical_variables'], uci_datasets_info[name]['id_variables'])
    tensor_data = torch.tensor(df.to_numpy(), dtype=torch.float32, device='cuda')

    col_widths = (tensor_data.max(dim=0).values + 1e-6 - tensor_data.min(dim=0).values) / num_bins
    tar = tensor_data[:, 0].unsqueeze(1).contiguous()
    pos = tensor_data[:, 1:].contiguous()

    hist_idx = (tar // col_widths[0]).long().squeeze(-1)
    hists = torch.zeros((df.shape[0], num_bins), device='cuda')
    hists[torch.arange(df.shape[0]), hist_idx] = 1.0
    global_hist = hists.sum(dim=0)

    init_best_nodes = results['best_detail']['best_nodes']
    # print(f'Init Number of Items = {hists[init_best_nodes].sum().item()}')

    pocket_details = {}
    with open(f"./perceptive_graphs/{name}_{results['pocket_size']}.pkl", 'rb') as file:
        pocket_details = pickle.load(file)

    pockets = pocket_details['pockets']
    c = pocket_details['midpoints']
    cl_hists = pocket_details['pocket_hists']

    clusters = results['best_detail']['clusters']
    sub_hist = cl_hists[init_best_nodes].sum(dim=0)
    
    print(results['best_config'], results['pocket_size'], len(results['best_detail']['best_nodes']))
    
    print(f'Starting Cardinality = {sub_hist.sum()}')
    print(f'Starting JSD = {compute_marginal_stat(default_base, sub_hist.unsqueeze(0), global_hist, size_corrected=False)[0].item()}')

    # desired_output =  (0.1 * pos.shape[0]) ** 0.5
    desired_stat = 1.0
    desired_size = 0.1 * pos.shape[0]
    desired_output = desired_stat * (desired_size ** 0.5)

    def fitness_func(ga_instance, solution, solution_idx):
        thresholds = [np.sqrt(chi2.ppf(perc, df=pos.shape[1])) for perc in solution]
        coverage_mask = get_confidence_interval_mask(pos, clusters, m_thresholds=thresholds)
        coverage_nodes = torch.nonzero(coverage_mask).squeeze(-1)
        sub_hist = hists[coverage_nodes].sum(dim=0)
        
        stat = compute_marginal_stat(default_base, sub_hist.unsqueeze(0), global_hist, size_corrected=False)[0].item()
        size = sub_hist.sum().item() ** 0.5

        if np.isnan(stat):
            print(solution)
            # print(coverage_nodes)
            # print(sub_hist)
        
        # score = 0.0 if np.isnan(score) else score
        
        # stat_fitness = 1.0 / (np.abs(stat - desired_stat) + 1e-10)
        # size_fitness = 1.0 / (np.abs(size - desired_size) + 1e-10)
        fitness = 1.0 / (np.abs(stat * size - desired_output) + 1e-6)
        # fitness = -1e-10 if np.isnan(fitness) else fitness

        # fitness = stat_fitness #* size_fitness
        
        return fitness

    num_generations = 100 # Number of generations.
    num_parents_mating = 50 # Number of solutions to be selected as parents in the mating pool.

    sol_per_pop = 250 # Number of solutions in the population.
    num_genes = len(clusters)

    print(f'Number of Genes = {num_genes}')

    ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        # mutation_type=None,
                        # crossover_type=None,
                        mutation_percent_genes="default" if num_genes > 10 else 100 // num_genes,
                        mutation_probability=0.75,
                        # keep_elitism=int(0.25 * sol_per_pop),
                        fitness_func=fitness_func,
                        on_generation=on_generation,
                        gene_type=float,
                        # parallel_processing=['process', 4],
                        gene_space={"low": 0, "high": 1})
    
    try:
        # Running the GA to optimize the parameters of the function.
        start = time.time()

        ga_instance.run()

        end = time.time()
        runtime = end - start

        # Returning the details of the best solution.
        solution, solution_fitness, solution_idx = ga_instance.best_solution(ga_instance.last_generation_fitness)

        if ga_instance.best_solution_generation != -1:
            print(f"Best fitness value reached after {ga_instance.best_solution_generation} generations.")

        print(f"Parameters of the best solution : {solution}")

        coverage_mask = get_confidence_interval_mask(pos, clusters, m_thresholds=solution)
        coverage_nodes = torch.nonzero(coverage_mask).squeeze(-1)
        print(f'Cardinality = {coverage_nodes.shape[0]}')
        sub_hist = hists[coverage_nodes].sum(dim=0).cpu().numpy()
            
        sub = sub_hist / sub_hist.sum()
        rem_hist = global_hist.cpu().numpy()
        rem = rem_hist / rem_hist.sum()

        M = 0.5 * (sub + rem)
        kl_p_m = np.sum(sub * np.log2((sub + 1e-10) / (M + 1e-10)))
        kl_q_m = np.sum(rem * np.log2((rem + 1e-10) / (M + 1e-10)))
        jsd = np.sqrt(0.5 * kl_p_m + 0.5 * kl_q_m) 

        print(f'JSD = {jsd}')

        with open(f'./results/genetic/{name}_genetic_{measure}.pkl', 'wb') as f:
            pickle.dump(
                {
                    'score': jsd,
                    'thresholds': solution,
                    'cardinality': coverage_nodes.shape[0],
                    'runtime': runtime
                }, f)
    except:
        print('Error occured !')


