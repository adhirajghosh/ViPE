#!/bin/bash
#SBATCH --job-name=rt-ds                # Job name
#SBATCH --partition=a100              # partition
#SBATCH --gres=gpu:1                  # type and number of gpus
#SBATCH --mem=300G                           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --nodes 1# number of nodes
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00                     # job will be cancelled after 6h 30min, max is 72h
#SBATCH --output=/mnt/lustre/lensch/lhr027/checkpoints/logs/ml_cloud/run-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=adhiraj.ghosh@student.uni-tuebingen.de
# insert your commands here
source "$HOME/.bashrc"  # Load your shell's configuration

conda activate /mnt/lustre/lensch/lhr027/.conda/envs/env

cd /mnt/lustre/lensch/lhr027/projects/SongAnimator/

srun  --ntasks-per-node=1 python new_evaluation/retrieval/retrieval_dataset.py --dataset ad_slogans --batch_size 10