import os

sizes = {
    "N": [16000, 32000, 64000, 128000, 256000, 512000, 1024000, 2048000, 4096000],
    "C": [16, 32, 64, 128, 256, 512, 1024]
}

durations = {
    16000: [0, 0, 30],
    32000: [0, 0, 30],
    64000: [0, 0, 30],
    128000: [0, 0, 30],
    256000: [0, 1, 0],
    512000: [0, 6, 0],
    1024000: [0, 8, 0]
}

# for name in ["N", "C"]:
for name in ["C"]:
    qos = ""

    for size in sizes[name]:
        if name == "N":
            server = 'gpu-l40s'
            days = durations[size][0]
            dur = durations[size][1]
            mins = durations[size][2]
            mem = 8192
        elif name == "C":
            server = 'gpu-l40s'
            # qos = "#SBATCH --qos=feit"
            # server = 'gpu-l40s'
            days = 0
            dur = 2
            mins = 0
            mem = 8192
        
        # if os.path.isfile(f'./results/mod/synthetic_{name}{size}.pkl'):
        #     continue

        print(f"{name}_{size}")
        slurm = open(f"{name}_{size}.slurm", "w")
        slurm.write(
f"""#!/bin/bash

# Partition for the job:
#SBATCH --partition={server}
{qos}

# Multithreaded (SMP) job: must run on one node 
#SBATCH --nodes=1

# The name of the job:
#SBATCH --job-name="{name}_{size}"

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
python synthetic.py {name} {size}

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s
"""
        )
        slurm.close()

        os.system(f"sbatch {name}_{size}.slurm")