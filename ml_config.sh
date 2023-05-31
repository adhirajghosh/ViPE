#!/bin/bash
#SBATCH --job-name=gpt2-medium                 # Job name
#SBATCH --partition=a100-fat                # partition
#SBATCH --gres=gpu:9                  # type and number of gpus
#SBATCH --mem=70G                           # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH --nodes 1# number of nodes
#SBATCH --cpus-per-task=8
#SBATCH --time=72:00:00                     # job will be cancelled after 6h 30min, max is 72h
#SBATCH --output=/mnt/lustre/lensch/hshahmohammadi86/checkpoints/logs/ml_cloud/run-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=hasan.karezan@gmail.com
# insert your commands here
source "$HOME/.bashrc"  # Load your shell's configuration

conda activate /mnt/lustre/lensch/hshahmohammadi86/.conda/envs/env

cd /mnt/lustre/lensch/hshahmohammadi86/projects/SongAnimator/lyrics_to_prompts/
srun  python training_ml.py --ml 1 --batch_size 30 --learning_rate 5e-5 --model_name gpt2-medium --context_length 5

