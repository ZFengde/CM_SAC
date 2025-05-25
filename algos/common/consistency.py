
import torch as th
import numpy as np
import torch.nn.functional as F
import math

from algos.common.helpers import append_dims, append_zero, mean_flat
from stable_baselines3.common.preprocessing import get_action_dim

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = th.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = th.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings

class Consistency_Model:

    def __init__(
        self,
        env,
        device,
        sigma_data: float = 1,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="karras",
        steps=40,
        ts=None,
        sampler="onestep", 
    ):
        self.env = env
        self.action_dim = get_action_dim(self.env.action_space)
        self.action_low, self.action_high = self.env.action_space.low[0], self.env.action_space.high[0]

        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.num_timesteps = 40

        self.device = device
        
        self.sampler = sampler
        self.steps = steps
        self.ts = [0, 20, 40]
        # this is for sample
        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings_for_boundary_condition(self, sigma): # get the values of c_...
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        """Constructs the noise schedule of Karras et al. (2022)."""
        ramp = th.linspace(0, 1, n)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas).to(device)

    def consistency_losses(
        self,
        model,
        x_start, # batch * action_dim, TODO
        num_scales,
        state=None, # batch * obs_dim
        target_model=None,
        noise=None,
        critic=None,
        clip_range=0.2):
        
        noise = th.randn_like(x_start) # make this noise a \nabla[Q(s, a)]
        dims = x_start.ndim

        def denoise_fn(x, t, state=None): # get sample from x_t, t and the conditions
            return self.denoise(model, x, t, state)[1]
            
        @th.no_grad()
        def target_denoise_fn(x, t, state=None):
            return self.denoise(target_model, x, t, state)[1]

        indices = th.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        ) # random 

        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        ) # t (t_n+1) and t2(t_n) here is randomly generated based on indices
        t = t**self.rho # t_n+1 

        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho # t_n
        
        x_t = x_start + noise * append_dims(t, dims) # basically is x_start + noise

        dropout_state = th.get_rng_state() # random number generator
        # TODO, here is the recovered output of the model
        distiller = denoise_fn(x_t, t, state) # # predicted target based on t = t_n+1

        x_t2 = x_start + noise * append_dims(t2, dims).detach()

        th.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state) # predicted target based on t2=t_n
        distiller_target = distiller_target.detach()
        
        state_rpt = th.repeat_interleave(state.unsqueeze(1), repeats=1, dim=1)
        multi_actions = self.batch_multi_sample(model=model, state=state_rpt)
        q_value = critic.q1_batch_forward(state_rpt, multi_actions)
        multi_q_losses = - q_value.mean()

        terms = {}
        terms["multi_q_losses"] = multi_q_losses
        terms["consistency_losses"] = multi_q_losses

        return terms

    def denoise(self, model, x_t, sigmas, state): # get clean output from x_t
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, state)

        denoised = c_out * model_output + c_skip * x_t 
        denoised = denoised.clamp(-1, 1)
        # since the maximum of c_out is 0.5, output from model is 1
        # then denoised in [-0.5, 0.5]
        return model_output, denoised
    
    def batch_denoise(self, model, x_t, sigmas, state): # get clean output from x_t
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        rescaled_t = 1000 * 0.25 * th.log(sigmas + 1e-44)
        model_output = model(c_in * x_t, rescaled_t, state)

        denoised = c_out * model_output + c_skip * x_t 
        denoised = denoised.clamp(-1, 1)
        # since the maximum of c_out is 0.5, output from model is 1
        # then denoised in [-0.5, 0.5]
        return model_output, denoised

    def sample(self, model, state): # get clean output from x_T
        x_0 = self.sample_onestep(model, state)
        return x_0
    
    def sample_onestep(self, model, state):
        x_T = th.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]]) # stands for sigma input
        return self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]
    
    def batch_multi_sample(self, model, state): # 
        x_T = th.randn((state.shape[0], state.shape[1], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]]) # stands for sigma input
        x_0 = self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]
        return x_0
    
    def sample_log_prob(self, model, state, N=10, sigma=0.1):
        num_scales = 40
        # Step 1: get target action x_0 (either sample or user-provided)
        x_T = th.randn((state.shape[0], self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]]) # stands for sigma input
        x_0 = self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]  # [B, action_dim]

        # Step 2: sample N actions under same state
        state_repeat = state.unsqueeze(1).repeat(1, N, 1)  # [B, N, obs_dim]
        x_0_repeat = x_0.unsqueeze(1).repeat(1, N, 1)  # [B, N, obs_dim]
        noise = th.randn_like(x_0_repeat) # make this noise a \nabla[Q(s, a)]
        dims = x_0_repeat.ndim
        indices = th.randint(
            0, num_scales - 1, (state_repeat.shape[0], state_repeat.shape[1],), device=x_0.device
        ) # random 
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        ) # t (t_n+1) and t2(t_n) here is randomly generated based on indices
        t = t**self.rho # t_n+1 
        x_t_batch = x_0_repeat + noise * append_dims(t, dims) # basically is x_start + noise

        s_in = x_T.new_ones([x_t_batch.shape[0], x_t_batch.shape[1]]) # stands for sigma input
        samples = self.batch_denoise(model, x_t_batch, t, state_repeat)[1]

        x_0_exp = x_0.unsqueeze(1).detach()  # [B, 1, action_dim]
        sq_dist = ((x_0_exp - samples) ** 2).sum(dim=2)  # [B, N]

        log_kernel = -sq_dist / (2 * sigma ** 2)
        log_prob = th.logsumexp(log_kernel, dim=1) - math.log(N)

        return x_0, log_prob  # log_prob: [B]
