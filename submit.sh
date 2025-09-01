#!/bin/bash
#SBATCH --job-name=diversity
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/home/jp7467/slurm_logs/%j_%x.out
#SBATCH --error=/gpfs/home/jp7467/slurm_logs/%j_%x.err
#SBATCH --export=ALL
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=8
#SBATCH --partition=a100_dev,a100_short,a100_long

conda activate
conda activate rl

pwd

python toy_example.py -m \
\
vocab_size=100 \
num_iterations=1001 save_every=250 \
pretrain_policy.enable=False \
reward_type="bimodal-halves-binary-10" \
loss_cfg.advantage_type="raw" \
loss_cfg.kl_grad_type=rev-vanilla,fwd-simple \
loss_cfg.entropy_grad_type="analytical" \
compute_kl_to_target=True \
entropy_coeff=0.0 \
kl_coeff=5e-1,1e-1,5e-2,1e-2,5e-3,1e-3,5e-4,1e-4,5e-5,1e-5,5e-6,1e-6 \
ref_pol_eps=1e-20 \
batch_size=32 \
learning_rate=5e-3 \
grad_clip=100.0 \
seed=10 \
device=0



echo "Run finished at: "
date
exit