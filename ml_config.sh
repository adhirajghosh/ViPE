 #!/bin/bash
#SBATCH --job-name=first_job_hsh                  # Job name
#SBATCH --partition=gpu-2080ti                # partition
#SBATCH --gres=gpu:1                   # type and number of gpus
#SBATCH --mem=50G                           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --nodes 1                           # number of nodes
#SBATCH --time=72:00:00                     # job will be cancelled after 6h 30min, max is 72h
#SBATCH --output=/mnt/qb/work2/lensch0/hshahmohammadi86/slurm/out/run-%j.out
#SBATCH --error=/mnt/qb/work2/lensch0/hshahmohammadi86/slurm/error/run-%j.err
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hasan.karezan@gmail.com
# insert your commands here
echo '---------------- Status of this machine: ----------------'
nvidia-smi
echo ""
# run training with all available GPUs
echo 'number of CPUs available to this job:'
echo $((SLURM_JOB_CPUS_PER_NODE))
echo '--'
# set this variable to fix nltk ssl problems
# export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
# debugging flags (optional)
# export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1
export TOKENIZERS_PARALLELISM="true"
# export CUDA_LAUNCH_BLOCKING=1
sleep 250000
echo '---------------- finished ----------------'
