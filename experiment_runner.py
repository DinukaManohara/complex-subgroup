import os
from datasets.metadata import uci_datasets_info

for name in uci_datasets_info:
    qos = ""

    if uci_datasets_info[name]['size'] <= 10000:
        server = 'gpu-a100-short'
        days = 0
        dur = 2
        mins = 30
        mem = 4096
        pocket_sizes = [4, 8, 16, 32]
    elif uci_datasets_info[name]['size'] <= 50000:
        server = 'gpu-l40s'
        # qos = "#SBATCH --qos=feit"
        # server = 'gpu-l40s'
        days = 0
        dur = 2
        mins = 30
        mem = 8192
        pocket_sizes = [8, 16, 32, 64]
    elif uci_datasets_info[name]['size'] <= 100000:
        server = 'gpu-h100'
        # qos = "#SBATCH --qos=feit"
        days = 0
        dur = 2
        mins = 30
        mem = 8192
        pocket_sizes = [16, 32, 64, 128]
    elif uci_datasets_info[name]['size'] <= 200000:
        server = 'gpu-l40s'
        days = 0
        dur = 4
        mins = 0
        mem = 8192
        pocket_sizes = [256, 512, 1024, 2048]

    for pocket_size in pocket_sizes:
        if os.path.isfile(f'./perceptive_graphs/{name}_{pocket_size}.pkl'):
            # is_all_done = True
            # for measure in ['jsd', 'kld', 'hellinger', 'tvd']:
            #     if not os.path.isfile(f'./results/mod/{name}_p{pocket_size}_{measure}.pkl'):
            #         is_all_done = False
            #         break
            
            # if is_all_done:
            #     continue

            for measure in ['jsd', 'kld', 'hellinger', 'tvd']:
                if os.path.isfile(f'./results/mod/{name}_p{pocket_size}_{measure}.pkl'):
                    print(f'Skipping {name}_p{pocket_size}_{measure}.')
                    continue

                print(f"{name}_p{pocket_size}")
                slurm = open(f"{name}_p{pocket_size}_{measure}.slurm", "w")
                slurm.write(
f"""#!/bin/bash

# Partition for the job:
#SBATCH --partition={server}
{qos}

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="{name}_p{pocket_size}_{measure}"

# The project ID which this job should run under:
#SBATCH --account="punim2258"

# Maximum number of tasks/CPU cores used by the job:
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Number of GPUs requested per node:
#SBATCH --gres=gpu:1
# The amount of memory in megabytes per node:
#SBATCH --mem={mem}

# Use this email address:
#SBATCH --mail-user=bdezoysa@student.unimelb.edu.au

# Send yourself an email when the job:
# aborts abnormally (fails)
#SBATCH --mail-type=FAIL
# begins
#SBATCH --mail-type=BEGIN
# ends successfully
#SBATCH --mail-type=END

# The maximum running time of the job in days-hours:mins:sec
#SBATCH --time={days}-{dur}:{mins}:00

# check that the script is launched with sbatch
if [ "x$SLURM_JOB_ID" == "x" ]; then
echo "You need to submit your job to the queuing system with sbatch"
exit 1
fi

# Run the job from the directory where it was launched (default)

# The modules to load:
module load GCCcore/11.3.0 CUDA/12.4.1 UCX-CUDA/1.16.0-CUDA-12.4.1 cuDNN/9.6.0.74-CUDA-12.4.1

# The job command(s):
python experiments.py {name} {pocket_size} {measure}

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
"""
                )
                slurm.close()

                os.system(f"sbatch {name}_p{pocket_size}_{measure}.slurm")