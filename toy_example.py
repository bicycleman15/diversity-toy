import logging
import os
from functools import partial
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt

import hydra
import numpy as np
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from metric_logger import Logger

hlog = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DecoderOnlyTransformer(nn.Module):
    """A simple decoder-only transformer for language modeling."""
    
    def __init__(self, vocab_size: int, d_model: int = 128, n_heads: int = 4, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding (simple learned positional embeddings)
        self.pos_embedding = nn.Embedding(1024, d_model)  # Max sequence length of 1024
        
        # Decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        
        # Output projection to logits
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len] -> [batch_size, seq_len, vocab_size]
        batch_size, seq_len = x.shape
        
        # Create position indices
        pos_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        
        # Get embeddings
        token_embeds = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        pos_embeds = self.pos_embedding(pos_indices)  # [batch_size, seq_len, d_model]
        
        # Combine token and positional embeddings
        embedded = token_embeds + pos_embeds
        
        # Create causal mask for decoder (lower triangular)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # Decoder forward pass
        # For decoder-only, we use the same input as both query and key/value
        decoded = self.decoder(
            tgt=embedded,
            memory=embedded,  # Self-attention only
            tgt_mask=causal_mask,
            memory_mask=None,
            tgt_key_padding_mask=None,
            memory_key_padding_mask=None
        )
        
        # Project to logits
        logits = self.output_proj(decoded)
        return logits
    

class CategoricalPolicy(nn.Module):
    """A simple policy that is parameterized as a categorical distribution over actions."""
    
    def __init__(self, vocab_size: int, init_logit_value: float = 0.0):
        super().__init__()
        self.vocab_size = vocab_size
        # Logits parameter for the categorical distribution
        self.logits = nn.Parameter(torch.ones(vocab_size) * init_logit_value)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len] - input tokens (ignored)
        
        Returns:
            log_probs: [batch_size, vocab_size] - log probabilities for next token
        """
        batch_size = x.shape[0]
        logits = einops.repeat(self.logits, "v -> b v", b=batch_size)  # [batch_size, vocab_size]
        return logits


class DummyReferencePolicy(nn.Module):
    """A dummy reference policy that stores hard-coded probabilities."""
    
    def __init__(self, vocab_size: int, ref_pol_eps: float):
        super().__init__()
        self.vocab_size = vocab_size
        # Create a hard-coded probability distribution
        # For example, favor lower token indices
        probs = torch.zeros(vocab_size)
        probs[:vocab_size//2] = 1.0  # Give higher probability to first half
        probs = probs + ref_pol_eps
        self.probs = probs / probs.sum()  # Normalize
        # NOTE: If any entry in self.probs is exactly 0, torch.log(0) -> -inf here.
        # This can propagate to NaN/inf downstream (e.g., MSE loss, KL terms).
        # Prefer setting a strictly positive `ref_pol_eps` in config to avoid exact zeros.
        self.log_probs = torch.log(self.probs)
    
    def __call__(self, input_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_tokens: [batch_size, seq_len] - input tokens (ignored)
        
        Returns:
            log_probs: [batch_size, vocab_size] - log probabilities for next token
        """
        batch_size = input_tokens.shape[0]
        # Return the same log probs for all samples in batch
        return self.log_probs.unsqueeze(0).expand(batch_size, -1).to(input_tokens.device)


def compute_rewards(actions: torch.Tensor, vocab_size: int, reward_type: str) -> torch.Tensor:
    """
    Compute rewards based on actions. This is an arbitrary reward function.
    
    Args:
        actions: [batch_size] - policy model actions
        vocab_size: Size of vocabulary
        reward_type: Type of reward function
    
    Returns:
        rewards: [batch_size] - reward for each sample
    """
    # Shared tiny helper for a single Gaussian on x in [0,1]
    def _gauss(x: torch.Tensor, mu: float, sigma: float = 0.03) -> torch.Tensor:
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    if reward_type == "linear":
        # Linear monotonically increasing reward from 0 to vocab_size - 1
        # Higher token indices get higher rewards
        rewards = actions.float() / vocab_size * 10  
    elif reward_type == "zero-one":
        # Reward is 1 if action is in 2nd or 4th quadrant of vocab, else 0
        quadrant = (actions * 4) // vocab_size  # 0, 1, 2, or 3
        is_second_or_fourth = (quadrant == 1) | (quadrant == 3)
        rewards = is_second_or_fourth.float()
    elif reward_type == "bimodal-split-halves":
         # Exactly two modes total: one in first half (amp 0.9), one in second (amp 1.0)
         denom = max(vocab_size - 1, 1)
         t = actions.float() / denom  # normalize to [0,1]
         rewards = 0.9 * _gauss(t, mu=0.2) + 1.0 * _gauss(t, mu=0.8)
    elif reward_type == "bimodal-both-firsthalf":
         # Exactly two modes, both centered in the first half (amps 0.9 and 1.0)
         denom = max(vocab_size - 1, 1)
         t = actions.float() / denom  # normalize to [0,1]
         # Centers at 1/8 and 3/8 keep them well inside the first half
         rewards = 0.9 * _gauss(t, mu=0.12) + 1.0 * _gauss(t, mu=0.38)
    elif reward_type == "bimodal-halves-binary-10":
         # Set 5 indices to 1 at the end of 1st quadrant and end of 3rd quadrant
         quadrant_size = vocab_size // 4
         first_quad = torch.arange(quadrant_size - 5, quadrant_size, device=actions.device)
         third_quad = torch.arange(3 * quadrant_size - 5, 3 * quadrant_size, device=actions.device)
         allowed = torch.cat([first_quad, third_quad])
         rewards = (actions.unsqueeze(1) == allowed.unsqueeze(0)).any(dim=1).to(dtype=torch.float32)
    elif reward_type == "bimodal-split-halves-equal":
         # Exactly two modes total with equal peak height 1.0 after discretization.
         # We snap the centers to the discrete grid and renormalize the summed curve
         # so that each peak attains exactly 1.0 on the grid (batch-independent).
         denom = max(vocab_size - 1, 1)
         t = actions.float() / denom  # normalize to [0,1]
         # Snap symmetric centers to grid indexes to preserve equality under discretization
         mu_left_idx = int(round(0.2 * denom))
         mu_right_idx = denom - mu_left_idx  # ensures symmetry w.r.t. the grid
         mu_left = mu_left_idx / denom
         mu_right = mu_right_idx / denom
         g_left = _gauss(t, mu=mu_left)
         g_right = _gauss(t, mu=mu_right)
         rewards_raw = g_left + g_right
         # Compute exact scalar normalization so the discrete peak value is 1.0
         center_left = t.new_tensor(mu_left)
         norm_const = 1.0 + _gauss(center_left, mu=mu_right)
         rewards = rewards_raw / norm_const
    
    else:
        raise NotImplementedError(f"Reward type {reward_type} not implemented")
    
    return rewards


def get_target_distribution(ref_probs: torch.Tensor, reward_vec: torch.Tensor, 
                            kl_coeff: float, ent_coeff: float):
    """
    Get the target distribution for KL regularization.

    Numerically stable log-space implementation.
    NOTE: If `ref_probs` contains exact zeros, log(0) = -inf and will carry through.
    Consider setting a positive epsilon in the reference policy to avoid exact zeros.
    """
    # Convert to torch.float32 tensors
    ref_probs_t = ref_probs.to(dtype=torch.float32)
    reward_vec_t = reward_vec.to(dtype=torch.float32)

    # Avoid divide-by-zero when coefficients are extremely small
    denom = ent_coeff + kl_coeff + 1e-10
    tempered_const = kl_coeff / denom

    # log g = tempered_const * log(ref) + reward/denom
    # log(ref) can be -inf if ref_probs has zeros
    log_ref = torch.log(ref_probs_t)
    log_unnorm = (tempered_const * log_ref) + (reward_vec_t / denom)
    logZ = torch.logsumexp(log_unnorm, dim=-1)
    normed_g = torch.exp(log_unnorm - logZ)
    return normed_g


def compute_kl_to_target(model: nn.Module, vocab_size: int, device: torch.device,
                           target_dist: torch.Tensor, temperature: float = 1.0
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes both forward and reverse KL divergence between the marginalized policy
    distribution and a target distribution.
    The marginalized distribution is computed by averaging the policy's output over all
    possible single-token inputs, weighted by a uniform prior over those inputs.
    This is a vectorized implementation.
    """
    with torch.no_grad():
        # Create a batch of all possible input tokens
        input_tokens = torch.arange(vocab_size, device=device).unsqueeze(1)  # [vocab_size, 1]

        # Get model output logits for all tokens at once
        logits = model(input_tokens)  # [vocab_size, 1, vocab_size]
        logits = logits.squeeze(1)  # [vocab_size, vocab_size]
        # NOTE: Ensure `temperature` > 0 when calling this function. Division by a non-positive
        # temperature can produce NaN/inf. We keep the code as-is to avoid implicit clamping.
        logits = logits / temperature

        # Convert to probabilities. Each row is the policy distribution for an input token.
        all_probs = F.softmax(logits, dim=-1)  # [vocab_size, vocab_size]

        # The input prior is uniform, so the marginalized distribution is the mean of all output distributions.
        marginalized_probs = all_probs.mean(dim=0)  # [vocab_size]

    # Let's denote P = marginalized_probs and Q = target_dist
    P = marginalized_probs
    Q = target_dist.to(device)

    # Add a small epsilon to avoid log(0)
    P = P.clamp(min=1e-10)
    Q = Q.clamp(min=1e-10)

    # Forward KL: KL(Q || P)
    fwd_kl = (Q * (torch.log(Q) - torch.log(P))).sum()

    # Reverse KL: KL(P || Q)
    rev_kl = (P * (torch.log(P) - torch.log(Q))).sum()

    return fwd_kl, rev_kl


def compute_marginalized_policy_distribution(
    model: nn.Module,
    vocab_size: int,
    device: torch.device,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Computes the policy distribution marginalized over a uniform prior on all
    single-token inputs. Returns a length-`vocab_size` probability vector.
    """
    with torch.no_grad():
        input_tokens = torch.arange(vocab_size, device=device).unsqueeze(1)
        logits = model(input_tokens).squeeze(1)
        # NOTE: Ensure `temperature` > 0 when calling this function. Division by a non-positive
        # temperature can produce NaN/inf. Kept as-is (no implicit clamping) for transparency.
        logits = logits / temperature
        all_probs = F.softmax(logits, dim=-1)
        marginalized_probs = all_probs.mean(dim=0)
    return marginalized_probs


def pretrain_policy_to_reference(
    policy_model: nn.Module,
    reference_policy: DummyReferencePolicy,
    cfg: DictConfig,
    device: torch.device,
    logger: Logger,
):
    """
    Pre-trains the policy model to match the reference policy.
    The goal is to minimize the MSE of log-probabilities between the two.
    """
    hlog.info("Starting pre-training of policy model...")
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=cfg.pretrain_policy.lr)

    policy_model.train()

    # Pre-fetch reference log probabilities (it's constant for this dummy policy)
    # We create a dummy input to get it on the right device.
    ref_log_probs_template = reference_policy(
        torch.empty(1, 1, dtype=torch.long, device=device)
    )  # [1, vocab_size], ref policy returns same log probs for all inputs

    for step in range(cfg.pretrain_policy.steps):
        # Sample dummy input tokens
        input_tokens = torch.randint(
            0, cfg.vocab_size, (cfg.batch_size, 1), device=device
        )  # [batch_size, 1]

        # Get log probabilities from the policy model
        policy_logits = policy_model(input_tokens).squeeze(1)  # [batch_size, vocab_size]
        policy_log_probs = F.log_softmax(policy_logits, dim=-1)  # [batch_size, vocab_size]

        # Expand reference log probs to match batch size
        ref_log_probs = ref_log_probs_template.expand_as(policy_log_probs)

        # Compute loss (MSE loss, not quite an KL but alas)
        # NOTE: If reference_policy produced any -inf in ref_log_probs (e.g., due to zero probs),
        # this MSE can become inf/NaN. Prefer strictly positive `ref_pol_eps` in config.
        loss = 0.5 * (policy_log_probs - ref_log_probs)**2
        loss = loss.mean()

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(),
                                                   cfg.grad_clip)
        optimizer.step()

        if step % cfg.pretrain_policy.log_every == 0:
            with torch.no_grad():
                # Compute KL to reference policy
                fwd_kl_ref, rev_kl_ref = compute_kl_to_target(
                    model=policy_model,
                    vocab_size=cfg.vocab_size,
                    device=device,
                    target_dist=reference_policy.probs.clone(),
                    temperature=1.0  # Default temperature
                )

                #hlog.info(
                #    f"[Pre-train] Step {step}/{cfg.pretrain_policy.steps}, Loss: {loss.item():.4f}"
                #)
                metrics = {
                    "pretrain/loss": loss.item(),
                    "pretrain/grad_norm": grad_norm.item(),
                    "pretrain/fwd_kl_ref": fwd_kl_ref.item(),
                    "pretrain/rev_kl_ref": rev_kl_ref.item(),
                    "pretrain/step": step,
                }
            logger.log(metrics, step=0)

    hlog.info("Finished pre-training.")
    return policy_model


def compute_pg_loss(data: Dict[str, torch.Tensor], policy_model: nn.Module, 
                    reference_policy: DummyReferencePolicy, kl_coeff: float, 
                    entropy_coeff: float, 
                    loss_cfg: DictConfig) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute total loss including policy gradient, KL regularization, and entropy regularization.
    
    Args:
        data: Dictionary containing training data with keys:
            - input_tokens: [batch_size, seq_len] - input tokens
            - actions: [batch_size] - sampled actions
            - action_log_probs: [batch_size] - log probabilities of sampled actions
            - rewards: [batch_size] - reward values
        policy_model: The policy model to compute current logits from
        reference_policy: Reference policy for KL regularization
        kl_coeff: KL penalty coefficient
        entropy_coeff: Entropy regularization coefficient
    
    Returns:
        total_loss: scalar tensor
        metrics: dict with all loss components
    """
    # Get current policy logits for the same inputs
    current_logits = policy_model(data["input_tokens"]).squeeze(1)  # [batch_size, vocab_size]
    log_probs = F.log_softmax(current_logits, dim=-1)  # [batch_size, vocab_size]
    pol_logps = log_probs.gather(1, data["actions"].unsqueeze(1)).squeeze(1)  # [batch_size]

    with torch.no_grad():
        ref_logps = reference_policy(data["input_tokens"])  # [batch_size, vocab_size]
        ref_logps = ref_logps.gather(1, data["actions"].unsqueeze(1)).squeeze(1)  # [batch_size]


    # Compute advantages (simple baseline: mean reward)
    if loss_cfg.advantage_type == "raw":
        advantages = data["rewards"]
    elif loss_cfg.advantage_type == "baseline_mean":
        advantages = data["rewards"] - data["rewards"].mean()
    elif loss_cfg.advantage_type == "baseline_mean_std":
        advantages = ((data["rewards"] - data["rewards"].mean()) 
                      / (data["rewards"].std() + 1e-8))
    else:
        raise NotImplementedError(f"Advantage type {loss_cfg.advantage_type} not implemented")

    policy_loss = -(pol_logps * advantages).mean()

    # KL gradient
    # NOTE: The forward-KL-style estimators below use exp(ref_logps - pol_logps).
    # If the log-ratio is large and positive, exp can overflow to inf before any
    # coefficient (e.g., `kl_coeff`) is applied. Consider clamping or alternative
    # formulations if you observe inf/NaN here when coefficients are very small.
    if loss_cfg.kl_grad_type in ("lowvar", "schulman-lowvar"):
        # Low var one proposed in http://joschu.net/blog/kl-approx.html
        # also as implemented in verl
        kl_penalty = torch.exp(ref_logps - pol_logps) - (ref_logps - pol_logps) - 1 
        kl_penalty = kl_penalty.mean()
    elif loss_cfg.kl_grad_type == "fwd-simple":
        # NOTE: this should yield exactly the same gradient as "schulman-lowvar" 
        # estimator which computes the forward KL gradient
        kl_penalty = pol_logps + torch.exp(ref_logps - pol_logps)
        kl_penalty = kl_penalty.mean()
    elif loss_cfg.kl_grad_type == "fwd-vanilla":
        # Simpliest vanilla version of the forward KL gradient
        kl_penalty = torch.exp(ref_logps - pol_logps)
        kl_penalty = kl_penalty.mean()
    elif loss_cfg.kl_grad_type == "fwd-mean-baseline":
        # Use cv_coef as the control variate, estimate it as mean of prob ratio
        with torch.no_grad():
            cv_coef = torch.exp(ref_logps - pol_logps).mean().detach()
        kl_penalty = (cv_coef * pol_logps) + torch.exp(ref_logps - pol_logps)
        kl_penalty = kl_penalty.mean()
    elif loss_cfg.kl_grad_type in ("rev-vanilla", "mse"):
        # Simple gradient for the reverse KL
        kl_penalty = 0.5 * (pol_logps - ref_logps)**2 
        kl_penalty = kl_penalty.mean()
    elif loss_cfg.kl_grad_type == "rev-mean-baseline":
        # Use estimated kl as the control variate
        with torch.no_grad():
            cv_coef = - (pol_logps - ref_logps).mean().detach()
        kl_penalty = 0.5 * (pol_logps - ref_logps)**2  + (cv_coef * pol_logps)
        kl_penalty = kl_penalty.mean()
    elif loss_cfg.kl_grad_type == "sym-kl-vanilla":
        # Simple gradient for symmetric KL
        kl_penalty_fwd = torch.exp(ref_logps - pol_logps)
        kl_penalty_rev = 0.5 * (pol_logps - ref_logps)**2 
        kl_penalty = 0.5 * (kl_penalty_fwd + kl_penalty_rev)
        kl_penalty = kl_penalty.mean()
    elif loss_cfg.kl_grad_type == "vanilla":
        kl_penalty = (pol_logps - ref_logps).mean()
    else:
        raise NotImplementedError(f"KL gradient type {loss_cfg.kl_grad_type} not implemented")

    # Entropy gradient
    if loss_cfg.entropy_grad_type == "analytical":
        ent_penalty = (log_probs * torch.exp(log_probs)).sum(dim=-1).mean()
    elif loss_cfg.entropy_grad_type == "vanilla":
        ent_penalty = pol_logps.mean()
    elif loss_cfg.entropy_grad_type == "mse":
        ent_penalty = 0.5 * (pol_logps ** 2).mean()
    else:
        raise NotImplementedError(f"Entropy gradient type {loss_cfg.entropy_grad_type} not implemented")

    # Total loss
    total_loss = policy_loss + (kl_coeff * kl_penalty) + (entropy_coeff * ent_penalty)


    # Metrics
    with torch.no_grad():
        # estimate entropy
        pol_entropy_esti = -pol_logps.mean()
        pol_entropy_analytical = - (log_probs * torch.exp(log_probs)).sum(dim=-1).mean()

        kl_estis_vanilla = pol_logps - ref_logps

        # Collect metrics
        metrics = {
            "policy_loss": policy_loss.item(),
            "kl_penalty": kl_penalty.item(),
            "entropy_penalty": ent_penalty.item(),
            "total_loss": total_loss.item(),
            "entropy_esti": pol_entropy_esti.item(),
            "entropy_analytical": pol_entropy_analytical.item(),
            "kl_esti_vanilla": kl_estis_vanilla.mean().item(),
            "kl_log_ratio_max": kl_estis_vanilla.max().item(),
            "reward_mean": data["rewards"].mean().item(),
            "reward_min": data["rewards"].min().item(),
            "reward_max": data["rewards"].max().item(),
            "advantage_mean": advantages.mean().item(),
            "advantage_min": advantages.min().item(),
            "advantage_max": advantages.max().item(),
        }
    
    return total_loss, metrics




@hydra.main(config_path=".", config_name="toy_config.yaml")
def main(CFG: DictConfig):
    # Resolve config
    CFG = OmegaConf.to_container(CFG, resolve=True)
    CFG = OmegaConf.create(CFG)
    hlog.info(f"Configuration: \n{OmegaConf.to_yaml(CFG)}")
    
    # Set seed for reproducibility
    set_seed(CFG.seed)
    
    # Setup
    device = torch.device(CFG.device)
    hlog.info(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = Path.cwd()
    hlog.info(f"Experiment directory: {exp_dir}")
    
    # Initialize logger
    logger = Logger(
        log_dir=exp_dir,
        use_wb=False,

        # wb_name=CFG.logger.wb_name,
        # wb_cfg=CFG
    )
    
    # Initialize models
    # policy_model = DecoderOnlyTransformer(
    #     vocab_size=CFG.vocab_size,
    #     d_model=CFG.d_model,
    #     n_heads=CFG.n_heads,
    #     n_layers=CFG.n_layers
    # ).to(device)
    policy_model = CategoricalPolicy(
        vocab_size=CFG.vocab_size,
        init_logit_value=CFG.policy_init_logit_value,
    ).to(device)
    
    # Dummy reference policy for KL regularization
    reference_policy = DummyReferencePolicy(
        vocab_size=CFG.vocab_size,
        ref_pol_eps=CFG.ref_pol_eps,
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=CFG.learning_rate)

    # Optionally pre-compute the target distribution
    target_dist_vec = None
    if CFG.compute_kl_to_target:
        refp_prob_vec = reference_policy.probs.clone().cpu()

        _reward_fn = partial(compute_rewards, vocab_size=CFG.vocab_size, 
                            reward_type=CFG.reward_type)
        reward_vec = _reward_fn(torch.arange(CFG.vocab_size))

        target_dist_vec = get_target_distribution(ref_probs=refp_prob_vec, 
                                                  reward_vec=reward_vec, 
                                                  kl_coeff=CFG.kl_coeff, 
                                                  ent_coeff=CFG.entropy_coeff)

    # Optionall pre-train the policy model toward the reference policy
    if CFG.pretrain_policy.enable:
        policy_model = pretrain_policy_to_reference(
            policy_model=policy_model,
            reference_policy=reference_policy,
            cfg=CFG,
            device=device,
            logger=logger,
        )
    
    # Training loop
    for iteration in range(CFG.num_iterations):
        hlog.info(f"Iteration {iteration}/{CFG.num_iterations}")
        
        # Generate target distributions for this batch
        batch_size = CFG.batch_size
        
        # Sample from policy model
        policy_model.eval()
        with torch.no_grad():
            # Generate sequences using the policy model
            # For simplicity, we'll generate single-token outputs
            # In practice, you'd want to generate full sequences
            
            # Create input prompts (could be random or fixed)
            input_tokens = torch.randint(0, CFG.vocab_size, (batch_size, 1)).to(device)
            
            # Get policy logits for the next token
            policy_logits = policy_model(input_tokens)  # [batch_size, 1]
            policy_logits = policy_logits.squeeze(1)  # [batch_size, vocab_size]
            
            # Sample actions from policy
            policy_probs = F.softmax(policy_logits, dim=-1)
            actions = torch.multinomial(policy_probs, 1).squeeze(-1)  # [batch_size]
            
            # Compute log probabilities of sampled actions
            log_probs = F.log_softmax(policy_logits, dim=-1)
            action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)  # [batch_size]
        
        # Compute rewards
        rewards = compute_rewards(actions, vocab_size=CFG.vocab_size, 
                                  reward_type=CFG.reward_type)

        data = {
            "input_tokens": input_tokens,
            "actions": actions,
            "action_log_probs": action_log_probs,
            "rewards": rewards,
        }

        # Compute policy loss with KL and entropy regularization
        policy_model.train()
        
        total_loss, loss_metrics = compute_pg_loss(
            data, policy_model, reference_policy,
            CFG.kl_coeff, CFG.entropy_coeff, CFG.loss_cfg
        )
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(policy_model.parameters(),
                                                   CFG.grad_clip)
        optimizer.step()
        
        # Logging
        metrics = {
            "iteration": iteration,
            "grad_norm": grad_norm.item(),
            **loss_metrics,
        }

        # Optionally compute KL to target distribution
        if CFG.compute_kl_to_target:
            fwd_kl_target, rev_kl_target = compute_kl_to_target(
                model=policy_model,
                vocab_size=CFG.vocab_size,
                device=device,
                target_dist=target_dist_vec,
                temperature=1.0  # Default temperature
            )
            metrics["fwd_kl_to_target"] = fwd_kl_target.item()
            metrics["rev_kl_to_target"] = rev_kl_target.item()

            fwd_kl_ref, rev_kl_ref = compute_kl_to_target(
                model=policy_model,
                vocab_size=CFG.vocab_size,
                device=device,
                target_dist=refp_prob_vec,
                temperature=1.0  # Default temperature
            )
            metrics["fwd_kl_to_ref"] = fwd_kl_ref.item()
            metrics["rev_kl_to_ref"] = rev_kl_ref.item()
        
        # Log metrics using the logger
        logger.log(metrics, step=iteration)
        
        # Print key metrics
        """
        if iteration % CFG.log_every == 0:
            hlog.info(f"Iteration {iteration}: Loss={total_loss.item():.4f}, "
                     f"Reward={loss_metrics['avg_reward']:.4f}, "
                     f"KL={loss_metrics['kl_div']:.4f}, "
                     f"Entropy={loss_metrics['entropy']:.4f}")
        """
        
        # Save checkpoint
        if iteration % CFG.save_every == 0 and iteration > 0:
            checkpoint = {
                "iteration": iteration,
                "policy_model_state_dict": policy_model.state_dict(),
                "reference_policy_probs": reference_policy.probs,
                "optimizer_state_dict": optimizer.state_dict(),
                "config": OmegaConf.to_container(CFG, resolve=True),
            }
            if CFG.save_policy_marginal:
                # Compute current marginalized policy distribution over tokens
                pol_marg_probs = compute_marginalized_policy_distribution(
                    model=policy_model,
                    vocab_size=CFG.vocab_size,
                    device=device,
                    temperature=1.0,
                ).cpu()
                checkpoint["policy_marginalized_probs_temp1"] = pol_marg_probs
            os.makedirs(exp_dir / "checkpoints", exist_ok=True)
            torch.save(checkpoint, exp_dir / f"checkpoints/ckpt_{iteration:06d}.pt")
            hlog.info(f"Saved checkpoint at iteration {iteration}")
    
    # Save final model
    #final_checkpoint = {
    #    "policy_model_state_dict": policy_model.state_dict(),
    #    "reference_policy_probs": reference_policy.probs,
    #    "config": OmegaConf.to_container(CFG, resolve=True),
    #}
    #torch.save(final_checkpoint, exp_dir / "final_model.pt")
    hlog.info("Training completed!")
    
    # Finish logging
    logger.finish()

    # give the final logits
    final_logits = policy_model.logits.detach().cpu()
    hlog.info(f"Final policy logits: {final_logits.numpy()}")
    hlog.info(f"Final policy probs: {F.softmax(final_logits, dim=-1).numpy()}")

    # plot the final policy probs
    plt.figure(figsize=(8, 4))
    plt.bar(np.arange(CFG.vocab_size), F.softmax(final_logits, dim=-1).numpy())
    plt.xlabel("Token Index")
    plt.ylabel("Probability")
    plt.title("Final Policy Token Distribution")
    plt.grid(True)
    plt.savefig(exp_dir / "final_policy_distribution.png")
    plt.close()
    hlog.info(f"Saved final policy distribution plot to: {exp_dir / 'final_policy_distribution.png'}")

    # now dump the path and some hyperparams to a csv file
    # hyperparams to write: seed, kl_coeff, entropy_coeff, reward_type, exp_dir / 'final_policy_distribution.png'
    # append to the csv file if it exists, else create it
    csv_path = Path("/gpfs/data/ranganathlab/Jatin/diversity") / "toy_example_results.csv"
    print(csv_path)

    header = "seed,kl_grad_type,kl_coeff,entropy_coeff,reward_type,final_policy_plot_path\n"
    row = f"{CFG.seed},{CFG.loss_cfg.kl_grad_type},{CFG.kl_coeff},{CFG.entropy_coeff},{CFG.reward_type},{exp_dir / 'final_policy_distribution.png'}\n"
    if not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write(header)
            f.write(row)
    else:
        with open(csv_path, "a") as f:
            f.write(row)
    print(row)


if __name__ == "__main__":
    main() 