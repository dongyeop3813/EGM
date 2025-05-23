import torch
import torch.nn.functional as F
import numpy as np
from flow_matching.path import MixtureDiscreteProbPath
from flow_matching.path.scheduler import ConvexScheduler, PolynomialConvexScheduler
from flow_matching.solver import MixtureDiscreteEulerSolver
from flow_matching.utils import ModelWrapper

from utils import next_time_step, log_mean_exp


class MaskedMixturePath(MixtureDiscreteProbPath):
    def __init__(self, scheduler: ConvexScheduler, num_tokens, seq_length):
        super().__init__(scheduler)
        self.num_tokens = num_tokens + 1  # 1 for mask
        self.seq_length = seq_length

    @property
    def mask(self):
        return self.num_tokens - 1

    def sample_x0(self, batch_size, device):
        return torch.full((batch_size, self.seq_length), self.mask, device=device)

    def p_t_given_r(
        self,
        x_t: torch.LongTensor,
        x_r: torch.LongTensor,
        t: torch.Tensor,
        r: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the backward transition probability p_{t|r}(x_t | x_r) for masked diffusion
        on a discrete state space, supporting batched inputs.

        Args:
            x_t: LongTensor of shape [..., seq_len] (token IDs at time t)
            x_r: LongTensor of shape [..., seq_len] (token IDs at time r)
            t: FloatTensor of shape [..., 1] (times t, broadcastable to x_t)
            r: FloatTensor of shape [..., 1] (times r, broadcastable)

        Returns:
            FloatTensor of shape [..., seq_len] with p_{t|r}(x_t^i| x_r^i).
        """
        # Compute schedule values
        alpha_t = self.scheduler(t).alpha_t  # [..., 1]
        alpha_r = self.scheduler(r).alpha_t  # [..., 1]
        ratio = alpha_t / alpha_r  # broadcastable to [..., seq_len]

        # Boolean masks
        is_mask_r = x_r == self.mask  # [..., seq_len]
        is_data_eq = x_t == x_r  # [..., seq_len]
        is_mask_t = x_t == self.mask  # [..., seq_len]

        # Case 1: x_r is mask -> stays mask
        prob_when_mask = is_mask_t.to(torch.float32)

        # Case 2: x_r is data symbol
        #   - if x_t == x_r: probability = ratio
        #   - elif x_t == mask: probability = 1 - ratio
        #   - else: 0
        prob_when_data = torch.where(
            is_data_eq,
            ratio,
            torch.where(is_mask_t, 1.0 - ratio, torch.zeros_like(ratio)),
        )

        # Combine based on whether x_r was mask
        prob = torch.where(is_mask_r, prob_when_mask, prob_when_data)
        return prob.prod(dim=-1)

    def p_t_given_1(self, xt, t, x1):
        """
        Compute the probability p_t(xt | x1) - the probability of xt at time t given x1.

        This method leverages the more general p_t_given_r method with r=1.

        Args:
            xt: Tensor of shape [..., L] representing token IDs at time t
            t: Tensor of shape [..., 1] representing time t
            x1: Tensor of shape [..., L] representing token IDs at time 1 (data)

        Returns:
            Tensor of shape [..., L] with probabilities p_t(xt | x1)
        """
        assert t.shape[-1] == 1
        assert (x1 != self.mask).all()

        # Set r=1 to compute p_t_given_1 using the general p_t_given_r method
        r = torch.ones_like(t)

        # Get probabilities from p_t_given_r
        prob = self.p_t_given_r(xt, x1, t, r)

        return prob

    def estimate(
        self,
        xt,
        t,
        energy,
        num_mc_samples=1000,
        proposal_type="naive",
        estimate_type="vel",
        **kwargs,
    ):
        """
        Estimate the conditional velocity or denoiser using Monte Carlo sampling.

        Args:
            xt: Tensor of shape [B, L] representing token IDs at time t
            t: Tensor of shape [B] representing time t
            energy: Energy function (computes log probability of samples)
            num_mc_samples: Number of Monte Carlo samples (default: 1000)
            proposal_type: Type of proposal distribution ("naive", "ais", "buffer", or "small_step")
            estimate_type: Type of estimation ("prob_vel", "prob_denoiser", or "bootstrap_denoiser")
            **kwargs: Additional arguments required for the chosen proposal_type

        Returns:
            est: Estimated conditional velocity or denoiser (shape [B, L, num_tokens])
        """
        # x1: [B, num_samples, L]
        # weight: [B, num_samples]
        x1, weight = self.proposal(
            proposal_type,
            xt,
            t,
            energy,
            num_mc_samples,
            **kwargs,
        )

        assert (
            weight.dim() == 2
            and weight.shape[1] == num_mc_samples
            and weight.shape[0] == x1.shape[0]
        ), f"weight.shape: {weight.shape}, x1.shape: {x1.shape}"

        assert (
            t.dim() == 1 and t.shape[0] == x1.shape[0]
        ), f"t.shape: {t.shape}, x1.shape: {x1.shape}"

        if estimate_type == "vel" and proposal_type == "naive":
            cond_vel = self.cond_vel(
                xt[:, None, :],
                t[:, None, None],
                x1,
            )

            est = (weight[..., None, None] * cond_vel).sum(dim=1)

        elif estimate_type == "vel" and proposal_type == "buffer":
            cond_vel = self.cond_vel(
                xt[:, None, :],
                t[:, None, None],
                x1,
            )

            est = (weight[..., None, None] * cond_vel).sum(dim=1)

        elif estimate_type == "denoiser" and proposal_type == "naive":
            cond_denoiser = self.cond_denoiser(
                xt[:, None, :],
                t[:, None, None],
                x1,
            )

            est = (weight[..., None, None] * cond_denoiser).sum(dim=1)

        elif estimate_type == "vel" and proposal_type == "small_step":
            xr = x1
            cond_vel = self.cond_vel_given_xr(
                xt[:, None, :],
                t[:, None, None],
                xr,
                kwargs["step_size"],
            )

            est = (weight[..., None, None] * cond_vel).sum(dim=1)

        elif estimate_type == "denoiser" and proposal_type == "small_step":
            xr = x1
            cond_denoiser = self.cond_denoiser_given_xr(
                xt[:, None, :],
                t[:, None, None],
                xr,
                kwargs["step_size"],
            )

            est = (weight[..., None, None] * cond_denoiser).sum(dim=1)

        elif estimate_type == "denoiser_with_flow" and proposal_type == "naive":
            cond_denoiser = self.cond_denoiser(
                xt[:, None, :],
                t[:, None, None],
                x1,
            )

            weight = torch.exp(-energy(x1))

            unnorm_flow = kwargs["flow_model"]

            est = (weight[..., None, None] * cond_denoiser).mean(dim=1)

            est = est / torch.exp(unnorm_flow(xt, t))[..., None, None]

        elif estimate_type == "denoiser_with_norm_flow" and proposal_type == "naive":
            cond_denoiser = self.cond_denoiser(
                xt[:, None, :],
                t[:, None, None],
                x1,
            )

            weight = torch.exp(-energy(x1))

            flow = kwargs["flow_model"]

            est = (weight[..., None, None] * cond_denoiser).mean(dim=1)

            est = (
                est
                / torch.exp(
                    flow(xt, t) - self.proposal_logZ(xt, t, torch.ones_like(t))
                )[..., None, None]
            )

        elif estimate_type == "denoiser_est_denom" and proposal_type == "naive":
            cond_denoiser = self.cond_denoiser(
                xt[:, None, :],
                t[:, None, None],
                x1,
            )

            weight = torch.exp(-energy(x1))

            est = (weight[..., None, None] * cond_denoiser).mean(dim=1)

            denom_x1, _ = self.proposal(
                proposal_type,
                xt,
                t,
                energy,
                num_mc_samples,
                **kwargs,
            )

            denom = torch.exp(-energy(denom_x1)).mean(dim=-1)

            est = est / denom[..., None, None]

        elif estimate_type == "self_denoiser" and proposal_type == "small_step":
            xr = x1
            prob_denoiser = kwargs["learned_prob_denoiser"]

            r = next_time_step(t, kwargs["step_size"])[:, None].expand(
                -1, num_mc_samples
            )
            is_one = r == 1.0

            v = prob_denoiser.prob_x1_given_xt(xr, r).detach()
            v[is_one] = F.one_hot(xr[is_one], num_classes=self.num_tokens).float()

            est = (weight[..., None, None] * v).sum(dim=1)

        elif estimate_type == "flow" and proposal_type == "naive":
            logits = -energy(x1)
            est = log_mean_exp(logits, dim=1)
            est += self.proposal_logZ(xt, t, torch.ones_like(t))

        elif estimate_type == "flow" and proposal_type == "small_step":
            xr = x1
            r = next_time_step(t, kwargs["step_size"])
            extended_r = r[:, None].expand(-1, num_mc_samples)

            is_one = extended_r == 1.0

            flow_model = kwargs["flow_model"]
            logits = flow_model(xr, extended_r).detach()

            logits[is_one] = -energy(xr[is_one])

            est = log_mean_exp(logits, dim=1)
            est += self.proposal_logZ(xt, t, r)

        else:
            raise ValueError(f"Invalid estimate type: {estimate_type}")

        return est

    def proposal(self, proposal_type, xt, t, energy, num_samples, **kwargs):
        if proposal_type == "naive":
            return self.naive_proposal(xt, t, energy, num_samples)

        elif proposal_type == "ais":
            return self.ais_proposal(
                xt,
                t,
                energy,
                num_samples,
                kwargs["num_steps"],
            )

        elif proposal_type == "buffer":
            return self.perturbed_buffer_proposal(
                xt,
                t,
                energy,
                kwargs["buffer_data"],
                num_samples,
                kwargs["noise_rate"],
            )

        elif proposal_type == "small_step":
            return self.small_step_proposal(
                xt,
                t,
                energy,
                kwargs["flow_model"],
                num_samples,
                kwargs["step_size"],
            )

        else:
            raise ValueError(f"Invalid proposal type: {proposal_type}")

    def cond_vel(self, xt, t, x1):
        """
        Compute the conditional velocity.

        Expected shape of input tensor:
        xt: [batch_dim..., L]
        t: [batch_dim..., 1]
        x1: [batch_dim..., L]

        Expected shape of the output tensor:
        cond_vel: [batch_dim..., L, num_tokens]
        """

        scheduler_output = self.scheduler(t)
        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t

        coeff = d_kappa_t / (1 - kappa_t)

        # coeff: [batch_dim..., 1] -> [batch_dim..., 1, 1]
        coeff = coeff[..., None]

        # one_hot_x1: [batch_dim..., L, num_tokens]
        one_hot_x1 = F.one_hot(x1, num_classes=self.num_tokens)
        one_hot_xt = F.one_hot(xt, num_classes=self.num_tokens)

        # cond_vel: [batch_dim..., L, num_tokens]
        cond_vel = coeff * (one_hot_x1 - one_hot_xt)

        return cond_vel

    def cond_vel_given_xr(self, xt, t, xr, step_size):
        """
        Compute the conditional velocity given xr.

        Args:
            xt: [batch_dim..., L]
            t: [batch_dim..., 1]
            xr: [batch_dim..., L]
            step_size: float

        Expected shape of the output tensor:
        cond_vel: [batch_dim..., L, num_tokens]
        """
        r = next_time_step(t, step_size)

        scheduler_output = self.scheduler(t)
        kappa_t = scheduler_output.alpha_t
        d_kappa_t = scheduler_output.d_alpha_t
        kappa_r = self.scheduler(r).alpha_t

        # coeff: [batch_dim..., 1] -> [batch_dim..., 1, 1]
        coeff = d_kappa_t / (kappa_r - kappa_t)
        coeff = coeff[..., None]

        # xr_one_hot: [batch_dim..., L, num_tokens]
        xr_one_hot = F.one_hot(xr, num_classes=self.num_tokens)

        # xt: [batch_dim..., L] -> [batch_dim..., 1, L]
        xt_one_hot = F.one_hot(xt, num_classes=self.num_tokens)

        # cond_vel: [batch_dim..., L, num_tokens]
        cond_vel = coeff * (xr_one_hot - xt_one_hot)

        return cond_vel

    def cond_denoiser(self, xt, t, x1):
        """
        Compute the conditional denoiser.

        Expected shape of input tensor:
        xt: [batch_dim..., L]
        t: [batch_dim..., 1]
        x1: [batch_dim..., L]

        Expected shape of the output tensor:
        cond_denoiser: [batch_dim..., L, num_tokens]
        """
        x1_one_hot = F.one_hot(x1, num_classes=self.num_tokens)
        return x1_one_hot

    def cond_denoiser_given_xr(self, xt, t, xr, step_size):
        """
        Compute the conditional denoiser given xr.

        Args:
            xt: [batch_dim..., L]
            t: [batch_dim..., 1]
            xr: [batch_dim..., L]
            step_size: float

        Expected shape of the output tensor:
        cond_denoiser: [batch_dim..., L, num_tokens]
        """
        r = next_time_step(t, step_size)

        kappa_t = self.scheduler(t).alpha_t
        kappa_r = self.scheduler(r).alpha_t

        # coeff_a: [batch_dim..., 1] -> [batch_dim..., 1, 1]
        coeff_a = (1 - kappa_t) / (kappa_r - kappa_t)
        coeff_a = coeff_a[..., None]

        # coeff_b: [batch_dim..., 1] -> [batch_dim..., 1, 1]
        coeff_b = (1 - kappa_r) / (kappa_r - kappa_t)
        coeff_b = coeff_b[..., None]

        # xr_one_hot: [batch_dim..., L, num_tokens]
        xr_one_hot = F.one_hot(xr, num_classes=self.num_tokens)

        # xt: [batch_dim..., L] -> [batch_dim..., 1, L]
        xt_one_hot = F.one_hot(xt, num_classes=self.num_tokens)

        # cond_denoiser: [batch_dim..., L, num_tokens]
        cond_denoiser = coeff_a * xr_one_hot - coeff_b * xt_one_hot

        return cond_denoiser

    def naive_proposal(self, xt, t, energy=None, num_samples=1000):
        is_filled = xt != self.mask

        B, L = xt.shape
        x1 = torch.randint(
            0,
            self.num_tokens - 1,
            (B, num_samples, L),
            device=xt.device,
        )

        # Expand xt to match dimensions with x1: [B, L] -> [B, num_samples, L]
        xt = xt.unsqueeze(1).expand(-1, num_samples, -1)

        # Expand is_filled mask: [B, L] -> [B, num_samples, L]
        is_filled = is_filled.unsqueeze(1).expand(-1, num_samples, -1)

        # Keep original values where not masked, use sampled values where masked
        x1 = torch.where(is_filled, xt, x1)

        if energy is not None:
            logits = -energy(x1)
            weight = torch.softmax(logits, dim=-1)
            return x1, weight
        else:
            return x1

    def ais_proposal(self, xt, t, energy, num_samples=1000, num_steps=10):
        """
        Annealed Importance Sampling proposal distribution.
        Samples are drawn from a distribution that interpolates between
        naive proposal and energy-based distribution using time t.

        Args:
            xt: [B, L] - current state
            t: [B] - time step
            energy: callable - energy function
            num_samples: int - number of samples to draw

        Returns:
            x1: [B, num_samples, L] - proposed samples
        """
        # Get naive proposal samples
        dt = 1.0 / num_steps
        B = xt.shape[0]

        x1 = self.naive_proposal(xt, t, num_samples)
        is_masked = xt == self.mask
        is_masked = is_masked.unsqueeze(1).expand(-1, num_samples, -1)

        def annealed_energy(x, i):
            t = i * dt
            return t * energy(x)

        def ais_step(prev_x, i):
            proposed_x = prev_x.clone()

            proposed_x[is_masked] = torch.randint(
                0, self.num_tokens - 1, (is_masked.sum(),), device=prev_x.device
            )

            log_accept_prob = -annealed_energy(proposed_x, i) + annealed_energy(
                prev_x, i
            )

            accept = torch.rand_like(log_accept_prob) < torch.exp(log_accept_prob)
            next_x = torch.where(accept.unsqueeze(-1), proposed_x, prev_x)

            log_weight = -annealed_energy(prev_x, i) + annealed_energy(next_x, i)

            return next_x, log_weight

        total_log_weight = torch.zeros(B, num_samples, device=xt.device)
        for i in range(num_steps):
            x1, log_w = ais_step(x1, i)
            total_log_weight += log_w

        total_log_weight += -energy(x1)

        weight = torch.softmax(total_log_weight, dim=-1)
        return x1, weight

    def perturbed_buffer_proposal(
        self,
        xt,
        t,
        energy,
        buffer_data,
        num_samples=1000,
        noise_rate=0.1,
    ):
        """
        xt: [B, L]
        t: [B]
        buffer_data: [K, L]
        """
        bsz, L = xt.shape

        def bit_flip_sample(data, noise_rate):
            """
            Bit flipping with probability noise_rate.

            data: [K, L]
            noise_rate: float

            return: [B, num_samples, L] - proposed samples
            """

            indices = torch.randint(len(data), (bsz, num_samples), device=data.device)
            x1 = data[indices]

            # Draw flip with probability noise_rate
            flip = torch.bernoulli(
                torch.ones(bsz, num_samples, L, device=data.device) * noise_rate
            )

            return ((x1 + flip) % 2).long()

        def bit_flip_log_prob(x, data, noise_rate):
            # xt: [..., L], data: [K, L] -> is_noised: [..., K, L]
            is_noised = x.unsqueeze(-2) != data.unsqueeze(0)

            p_each_dim = is_noised * noise_rate + ~is_noised * (1 - noise_rate)
            log_p_each_dim = torch.log(p_each_dim)
            log_p_each_mode = log_p_each_dim.sum(dim=-1)
            log_prob = torch.logsumexp(log_p_each_mode, dim=-1)

            return log_prob

        x1 = bit_flip_sample(buffer_data, noise_rate)
        log_prob_proposal = bit_flip_log_prob(x1, buffer_data, noise_rate)

        p_t_given_1 = self.p_t_given_1(
            xt[:, None, :].expand(-1, num_samples, -1),
            t[:, None, None].expand(-1, num_samples, -1),
            x1,
        )

        log_weight = -energy(x1) - log_prob_proposal

        weight = torch.exp(log_weight) * p_t_given_1
        weight /= weight.sum(dim=-1, keepdim=True)

        return x1, weight

    def small_step_proposal(
        self,
        xt,
        t,
        energy=None,
        flow_model=None,
        num_samples=1000,
        step_size=1e-2,
    ):
        """
        Generate samples from q_{r|t}(x_r| x_t) \propto p_{t|r}(x_t | x_r).
        Here, r is set to t + step_size.

        Args:
            xt: [B, L] - tokens at current time t
            t: [B] - current time
            energy: energy function
            flow_model: flow model
            num_samples: number of samples to generate
            step_size: time step size

        Returns:
            x1: [B, num_samples, L] - proposed samples
            weight: [B, num_samples] - weights for each sample
        """
        assert xt.dim() == 2 and t.dim() == 1

        B, L = xt.shape
        xt = xt.unsqueeze(1).expand(B, num_samples, L)

        r = next_time_step(t, step_size)

        t = t.unsqueeze(-1).expand(-1, num_samples)
        r = r.unsqueeze(-1).expand(-1, num_samples)
        is_one = r == 1.0

        alpha_t = self.scheduler(t).alpha_t
        alpha_r = self.scheduler(r).alpha_t
        ratio = alpha_t / alpha_r

        # Number of data tokens (excluding mask)
        num_data_tokens = self.num_tokens - 1

        # Compute probability of x_r stays to the mask when x_t = mask
        # p_stay_mask: [B, num_samples]
        p_stay_mask = 1.0 / (1.0 + num_data_tokens * (1.0 - ratio))
        p_stay_mask[is_one] = 0.0

        # Initialize output with x_t (if x_t != mask, x_r = x_t)
        xr = xt.clone()

        # Identify positions where x_t is mask
        is_mask_t = xt == self.mask

        # Decide which masked positions become data
        u = torch.rand(B, num_samples, L, device=xt.device)

        to_data = is_mask_t & (u >= p_stay_mask.unsqueeze(-1))

        # For positions in `to_data`, sample a random data token ID uniformly
        # Sample indices in [0, num_data_tokens)
        rand_idx = torch.randint(
            low=0,
            high=num_data_tokens,
            size=(B, num_samples, L),
            device=xt.device,
        )

        # Assign sampled data IDs
        xr[to_data] = rand_idx[to_data]

        if flow_model is not None:
            weight = torch.softmax(flow_model(xr, r), dim=-1)
            return xr, weight.detach()
        else:
            return xr

    def proposal_logZ(self, xt, t, r):
        """
        Compute the partition function for the proposal q_{r|t}(x_r| x_t).

        Z_{r|t}(x_t) = \sum_{x_r} p_{t|r}(x_t|x_r)

        Args:
            xt: [..., L] - tokens at current time t
            t: [...] - current time
            r: [...] - proposed time

        Returns:
            logZ: [...] - log partition function
        """
        alpha_t = self.scheduler(t).alpha_t
        alpha_r = self.scheduler(r).alpha_t
        ratio = (alpha_t / alpha_r).unsqueeze(-1)

        is_one = r == 1.0

        # [B, L]
        is_mask_t = xt == self.mask

        logZ = torch.log(ratio + (self.num_tokens) * (1 - ratio) * is_mask_t)
        logZ = torch.where(
            is_one.unsqueeze(-1),  # if r == 1.0
            torch.where(
                is_mask_t,
                # if mask, Z = d * (1 - kappa_t)
                torch.log((self.num_tokens - 1) * (1 - alpha_t)).unsqueeze(-1),
                # if data, Z = kappa_t
                torch.log(alpha_t).unsqueeze(-1),
            ),
            logZ,
        )
        return logZ.sum(dim=-1)


class WrappedModel(ModelWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, **extras):
        return self.model.prob_x1_given_xt(x, t)


def construct_masked_diffusion(
    prob_denoiser,
    num_tokens,
    seq_length,
    scheduler=PolynomialConvexScheduler(n=1.0),
):

    mixture_path = MaskedMixturePath(
        scheduler=scheduler,
        num_tokens=num_tokens,
        seq_length=seq_length,
    )

    wrapped_probability_denoiser = WrappedModel(prob_denoiser)

    solver = MixtureDiscreteEulerSolver(
        model=wrapped_probability_denoiser,
        path=mixture_path,
        vocabulary_size=num_tokens + 1,
    )

    return mixture_path, solver
