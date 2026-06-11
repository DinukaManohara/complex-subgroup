import os

qos = ""

server = 'gpu-h100'
mem = 16384

server_configs = {
    256:[2, 12, 0],
    512:[1, 12, 0],
    # 1024:[0, 18, 0],
    # 2048:[1, 12, 0]
}

pocket_sizes = [256, 512]#, 1024, 2048]

for pocket_size in pocket_sizes:
    days = server_configs[pocket_size][0]
    dur = server_configs[pocket_size][1]
    mins = server_configs[pocket_size][2]
    
    if os.path.isfile(f'./perceptive_graphs/case_study_{pocket_size}.pkl'):
        for measure in ['jsd']:
            if os.path.isfile(f'./results/mod/case_study_p{pocket_size}.pkl'):
                # print(f'Skipping {name}_p{pocket_size}.')
                continue

            print(f"case_study_p{pocket_size}")
            slurm = open(f"case_study_p{pocket_size}.slurm", "w")
            slurm.write(
f"""#!/bin/bash

# Partition for the job:
#SBATCH --partition={server}
{qos}

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="case_study_p{pocket_size}"

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
python case_study.py {pocket_size}

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
"""
            )
            slurm.close()

            os.system(f"sbatch case_study_p{pocket_size}.slurm")