#!/bin/bash
#SBATCH --job-name=g1                 # Job name
#SBATCH --partition=a100              # partition
#SBATCH --gres=gpu:1                  # type and number of gpus
#SBATCH --mem=100G                           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --nodes 1# number of nodes
#SBATCH --cpus-per-task=16
#SBATCH --time=72:00:00                     # job will be cancelled after 6h 30min, max is 72h
#SBATCH --output=/mnt/lustre/lensch/hshahmohammadi86/checkpoints/logs/ml_cloud/run-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hasan.karezan@gmail.com
# insert your commands here
source "$HOME/.bashrc"  # Load your shell's configuration

conda activate /mnt/lustre/lensch/hshahmohammadi86/.conda/envs/env

cd /mnt/lustre/lensch/hshahmohammadi86/projects/SongAnimator/new_evaluation/flute_haivmet/
srun  --ntasks-per-node=1 python image_genation_ml.py --batch_size 10 --model_name gpt2 --context_length 7
