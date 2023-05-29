#!/bin/bash
#SBATCH --job-name=gpt2_cxs                  # Job name
#SBATCH --partition=a100                # partition
#SBATCH --gres=gpu:8                  # type and number of gpus
#SBATCH --mem=50G                           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --nodes 1# number of nodes
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00                     # job will be cancelled after 6h 30min, max is 72h
#SBATCH --output=/mnt/lustre/lensch/hshahmohammadi86/checkpoints/logs/ml_cloud/run_%A_%a.out
#SBATCH --array=0-1%1
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hasan.karezan@gmail.com
# insert your commands here
source "$HOME/.bashrc"  # Load your shell's configuration

conda activate /mnt/lustre/lensch/hshahmohammadi86/.conda/envs/env

cd /mnt/lustre/lensch/hshahmohammadi86/projects/SongAnimator/lyrics_to_prompts/

#lrs=(2e-5 5e-5 2e-4 5e-4)

# Get the learning rate for the current job index
#lr=${lrs[$SLURM_ARRAY_TASK_ID]}
context_sizes=(1 3 5 7)
cxs=${context_sizes[$SLURM_ARRAY_TASK_ID]}

# Get the learning rate for the current job index
#lr=${lrs[$SLURM_ARRAY_TASK_ID]}

srun  python training_ml.py --ml 1  --batch_size 50 --learning_rate 2e-4 --context_length "$cxs"

#echo '---------------- Status of this machine: ----------------'
#nvidia-smi
#echo ""
## run training with all available GPUs
#echo 'number of CPUs available to this job:'
#echo $((SLURM_JOB_CPUS_PER_NODE))
#echo '--'
## set this variable to fix nltk ssl problems
## export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
## debugging flags (optional)
## export NCCL_DEBUG=INFO
#export PYTHONFAULTHANDLER=1
#export TOKENIZERS_PARALLELISM="true"
## export CUDA_LAUNCH_BLOCKING=1
#sleep 250000
#echo '---------------- finished ----------------'
