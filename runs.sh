python toy_example.py \
vocab_size=100 \
num_iterations=1001 save_every=250 \
pretrain_policy.enable=False \
reward_type="bimodal-halves-binary-10" \
loss_cfg.advantage_type="raw" \
loss_cfg.kl_grad_type="rev-vanilla" \
loss_cfg.entropy_grad_type="analytical" \
compute_kl_to_target=True \
entropy_coeff=0.0 \
kl_coeff=1e-1 \
ref_pol_eps=1e-20 \
batch_size=32 \
learning_rate=5e-3 \
grad_clip=100.0 \
seed=10 \
device=1

rev-vanilla
fwd-simple