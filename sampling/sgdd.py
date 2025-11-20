from .base import Algo
import torch
import torch.nn.functional as F
import tqdm
import numpy as np
from .sampling_utils import get_pc_sampler
import os
import pdb

class SGDD(Algo):
    """
        Implementation of split Gibbs sampling for discrete diffusion.
        https://arxiv.org/abs/2405.18782 (continuous version)
    """

    def __init__(self, net, forward_op, num_steps=50, ode_steps=32, eps=1e-5, mh_steps=200, alpha=30, max_dist = 1, device='cuda', cf=False):
        """
            Initializes the DAPS sampler with the given configurations.

            Parameters:
                annealing_scheduler_config (dict): Configuration for annealing scheduler.
                diffusion_scheduler_config (dict): Configuration for diffusion scheduler.
                lgvd_config (dict): Configuration for Langevin dynamics.
        """
        super().__init__(net=net, forward_op=forward_op)

        # self.model = self.net.model
        self.graph = self.net.graph
        self.noise = self.net.noise
        # self.uncond_sampler = get_pc_sampler(self.graph, self.noise, (1,1024), 'analytic', ode_steps , device=device)
        self.uncond_sampler = get_pc_sampler(self.graph, self.noise, (1,self.net.length), 'analytic', ode_steps , device=device)
        self.device = device
        self.num_steps = num_steps
        self.sigma_fn = lambda t: t
        self.get_time_step_fn = lambda r: (1 + r * (eps  - 1))
        steps = torch.linspace(0, 1-eps, num_steps)
        # steps = torch.linspace(0.5, 0.7, num_steps)
        # p = 0.245 # sigma=1
        # p = 0.302 # sigma=0.5
        # p = 0.379   # sigma=0.2
        # p = 0.434   # sigma=0.1
        # p = 0.49  # sigma=0.05
        # p = 0.61  # sigma=0.01
        # p = 0.68  # sigma=0.005
        # p = 0.8  # sigma=0.001
        # steps = torch.ones(num_steps) * p # for ablation study!
        # print(self.noise(1-p)[0])
        
        self.time_steps = self.get_time_step_fn(steps)
        self.ode_steps = ode_steps
        self.mh_steps = mh_steps
        self.alpha = alpha
        self.max_dist = max_dist
        self.cf = cf

    def log_ratio(self, sigma, hm_dist):
        
        alpha = (1 - np.exp(-sigma)) * (1 - 1/self.graph.dim)
        log_alpha = np.log(alpha+1e-5)
        log_1alpha = np.log(1 - alpha)
        log_ratio = hm_dist * log_alpha + (self.net.length - hm_dist) * log_1alpha
        return log_ratio
    
    def metropolis_hasting(self, x0hat, op, y, sigma, steps):
        x = x0hat.clone()
        dim = self.graph._dim
        N, L = x0hat.shape[0], x0hat.shape[1]
        # 여기서 log likelihood 계산
        current_log_likelihood = op.log_likelihood(x, y)
        current_hm_dist = (x != x0hat).sum(dim=-1)
        for _ in range(steps):

            # Get proposal
            
            for _ in range(self.max_dist):
                proposal = x.clone() # proposal, shape = [N, L]
                # for _ in range(self.max_dist):
                idx = torch.randint(L, (N,), device=x.device)
                v = torch.randint(dim, (N,), device=x.device)
                proposal.scatter_(1, idx[:, None], v.unsqueeze(1))

            # Compute log prob difference
            log_likelihood = op.log_likelihood(proposal,y)
            hm_dist = (proposal != x0hat).sum(dim=-1)
            log_ratio = log_likelihood - current_log_likelihood
            log_ratio += self.log_ratio(sigma, hm_dist) - self.log_ratio(sigma, current_hm_dist)

            # Metropolis-Hasting step
            rho = torch.clip(torch.exp(log_ratio), max=1.0)
            seed = torch.rand_like(rho)
            x = x * (seed > rho).unsqueeze(-1) + proposal * (seed < rho).unsqueeze(-1)
            current_log_likelihood = log_likelihood * (seed < rho)+ current_log_likelihood * (seed > rho)
            current_hm_dist = hm_dist * (seed < rho) + current_hm_dist * (seed > rho)
            
        return x

    @torch.no_grad()
    def inference(self, observation=None, num_samples=1, verbose=True, reward_model=None, eval_reward_model=None):
        """
        Modified inference to return values compatible with controlled_decode_rl.
        
        Returns:
            gen_samples: generated DNA samples tensor [num_samples, seq_len]
            zero_shot_gen_samples: baseline samples tensor [num_samples, seq_len]
            value_func_preds: placeholder tensor (not used in SGDD)
            reward_model_preds: reward predictions for generated samples
            eval_reward_model_preds: eval reward predictions for generated samples
            selected_baseline_preds: top k baseline predictions
            baseline_preds: all baseline predictions
            eval_base_reward_model_preds: eval reward predictions for baseline
            q_xs_history: list of q_xs distributions
            x_history: list of x states
            q_x0_history: list of predicted x0
            last_x_list: list of last_x values (one-hot encoded states)
        """
        
        pbar = tqdm.trange(self.num_steps) if verbose else range(self.num_steps)
        x_start = self.graph.sample_limit(num_samples, self.net.length).to(self.device)
        
        xt = x_start.to(self.device)
        
        # Track history for compatibility with controlled_decode_rl
        x0hats = []
        xts = []
        q_xs_history = []
        x_history = []
        q_x0_history = []
        last_x_list = []
        
        for i in pbar:
            # Store current state
            x_history.append(xt.clone())

            # 1. reverse diffusion
            # pdb.set_trace()
            x0hat = self.uncond_sampler(self.net, xt, t_start=self.time_steps[i])
            x0hats.append(x0hat.clone())
            
            # 2. Metropolis-Hasting
            sigma, _ = self.noise(self.time_steps[i])
            if self.cf:
                x0y = self.forward_op.cf_sample(x0hat, observation, sigma*self.alpha)
            else:
                # print(f'observation: {observation}')
                x0y = self.metropolis_hasting(x0hat, self.forward_op, observation, sigma*self.alpha, steps=self.mh_steps)
            xt = x0y
            xts.append(xt.clone())
            
            # Approximate q_xs as one-hot for compatibility
            # In discrete diffusion, q_xs represents p(x_{t-1}|x_t)
            # For SGDD we approximate it as delta distribution on sampled x
            q_xs_approx = F.one_hot(xt, num_classes=self.graph.dim).float()
            q_xs_history.append(q_xs_approx)
            
            # q_x0 is the predicted x0 distribution
            q_x0_approx = F.one_hot(x0hat, num_classes=self.graph.dim).float()
            q_x0_history.append(q_x0_approx)
            
            # last_x is one-hot encoded current state
            last_x = F.one_hot(xt, num_classes=self.graph.dim).float()
            last_x_list.append(last_x)
        
        # Final samples from SGDD (keep as tensor)
        gen_samples = xt
        
        # Generate baseline samples (unconditional)
        x_baseline = self.graph.sample_limit(num_samples, self.net.length).to(self.device)
        for i in range(self.num_steps):
            x0hat_base = self.uncond_sampler(self.net, x_baseline, t_start=self.time_steps[i])
            sigma, _ = self.noise(self.time_steps[i])
            # Unconditional: just use predicted x0
            x_baseline = x0hat_base
        zero_shot_gen_samples = x_baseline
        
        # Compute rewards if reward models are provided
        if reward_model is not None:
            # Transform samples for reward model
            onehot_samples = self.transform_samples_for_reward(xt)
            reward_model_preds = reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            onehot_baseline = self.transform_samples_for_reward(x_baseline)
            baseline_preds = reward_model(onehot_baseline.float().transpose(1, 2)).detach()[:, 0]
            selected_baseline_preds = baseline_preds  # Use all baseline preds
        else:
            reward_model_preds = torch.zeros(num_samples, device=self.device)
            baseline_preds = torch.zeros(num_samples, device=self.device)
            selected_baseline_preds = baseline_preds
        
        if eval_reward_model is not None:
            onehot_samples = self.transform_samples_for_reward(xt)
            eval_reward_model_preds = eval_reward_model(onehot_samples.float().transpose(1, 2)).detach()[:, 0]
            
            onehot_baseline = self.transform_samples_for_reward(x_baseline)
            eval_base_reward_model_preds = eval_reward_model(onehot_baseline.float().transpose(1, 2)).detach()[:, 0]
        else:
            eval_reward_model_preds = torch.zeros(num_samples, device=self.device)
            eval_base_reward_model_preds = torch.zeros(num_samples, device=self.device)
        
        # Value function predictions - placeholder
        value_func_preds = torch.zeros(num_samples, device=self.device)
        
        return (gen_samples, zero_shot_gen_samples, value_func_preds, 
                reward_model_preds, eval_reward_model_preds,
                selected_baseline_preds, baseline_preds, eval_base_reward_model_preds,
                q_xs_history, x_history, q_x0_history, last_x_list)
    
    def transform_samples_for_reward(self, samples, num_classes=4):
        """Transform samples to one-hot format for reward model."""
        # Mask out invalid tokens (assuming 4 is mask token)
        mask = samples != 4
        valid_samples = samples * mask
        one_hot_samples = F.one_hot(valid_samples, num_classes=num_classes)
        # Apply mask to zero out invalid rows
        one_hot_samples = one_hot_samples * mask.unsqueeze(-1)
        return one_hot_samples
    
    
# class SGDD_latent(SGDD):
#     def metropolis_hasting(self, x0hat, op, y, sigma, steps):
#         x = x0hat.clone()
#         dim = self.graph._dim
#         N, L = x0hat.shape[0], x0hat.shape[1]
#         ## decode x:
#         x_decoded = self.net.decode(x)
#         current_log_likelihood = op.log_likelihood(x_decoded, y)
#         current_hm_dist = (x != x0hat).sum(dim=-1)
#         for _ in range(steps):
#             for _ in range(self.max_dist):
#                 proposal = x.clone() # proposal, shape = [N, L]
#                 # for _ in range(self.max_dist):
#                 idx = torch.randint(L, (N,), device=x.device)
#                 v = torch.randint(dim, (N,), device=x.device)
#                 proposal.scatter_(1, idx[:, None], v.unsqueeze(1))
#             proposal_decoded = self.net.decode(proposal)
#             log_likelihood = op.log_likelihood(proposal_decoded,y)
#             hm_dist = (proposal != x0hat).sum(dim=-1)
#             log_ratio = log_likelihood - current_log_likelihood
#             log_ratio += self.log_ratio(sigma, hm_dist) - self.log_ratio(sigma, current_hm_dist)
#             rho = torch.clip(torch.exp(log_ratio), max=1.0)
#             seed = torch.rand_like(rho)
#             x = x * (seed > rho).unsqueeze(-1) + proposal * (seed < rho).unsqueeze(-1)
#             current_log_likelihood = log_likelihood * (seed < rho)+ current_log_likelihood * (seed > rho)
#             current_hm_dist = hm_dist * (seed < rho) + current_hm_dist * (seed > rho)
            
#         return x
    
#     def inference(self, observation=None, num_samples=1, verbose=True):
#         z = super().inference(observation, num_samples, verbose)
#         return self.net.decode(z)