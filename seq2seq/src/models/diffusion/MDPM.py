"""
Masked Diffusion Polymer Model (MDPM)
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, os.path.dirname(project_root))

from src.models.diffusion.layers import PolyDiffusionTransformer
from src.models.diffusion.schedulers import MaskedDiffusionSchedule


class MDPM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        mask_id: int,  # should we be transfering the tokenizer itself to get these ids ?
        pad_id: int,
        T: int = 1000,
        dim: int = 512,
        depth: int = 8,
        n_heads: int = 8,
        max_len: int = 700,
        dropout: float = 0.1,
        cond_dim=None,
    ):
        super().__init__()
        self.mask_id = mask_id
        self.pad_id = pad_id
        self.T = T

        self.scheduler = MaskedDiffusionSchedule(T=self.T)

        self.denoiser = PolyDiffusionTransformer(
            vocab_size=vocab_size,
            max_len=max_len,
            depth=depth,
            n_heads=n_heads,
            dropout=dropout,
            cond_dim=cond_dim,
        )

    def forward(self, x0, cond=None):
        B, L = x0.shape
        device = x0.device

        # samples random time step in range 1 and self.T to fill the
        # length of batch-size
        t_int = torch.randint(1, self.T, (B,), device=device)

        # setting scheduler with the input and t_int
        xt = self.scheduler.q_sample(x0, t_int, self.mask_id)

        # padding mask if the element is not equal to pad_id
        pad_mask = x0 != self.pad_id

        # forward pass through the denoiser
        logits = self.denoiser(xt, t_int, mask=pad_mask, cond=cond)

        # getting logical mask if the elemet is masked and is not in the padding
        is_masked = (xt == self.mask_id) & pad_mask

        # if there is not masked element just return 0 as loss
        if is_masked.sum() == 0:
            return torch.tensor(0.0, requires_grad=True, device=device)

        logits_flat = logits[is_masked]
        targets_flat = x0[is_masked]

        # getting the cross_entropy loss
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction="none")

        # get the scheduler weight for the time step
        weights = self.scheduler.get_loss_weight(t_int)

        # filter the loss for the predicted tokens only
        weights_flat = weights.unsqueeze(1).expand_as(is_masked)[is_masked]

        # computing final loss as CE * weights
        loss = (ce_loss * weights_flat).mean()
        return loss

    @torch.no_grad()
    def sample(self, B, L, device, cond=None, n_steps=100, temparature=1.0, top_p=0.9):
        # create a tensor of shape (B, L) filled with mask_id to start with
        xt = torch.full((B, L), self.mask_id, dtype=torch.long, device=device)

        # initialize the timesteps tensor where we reduce the timesteps from scheduler
        # to 1 by the step of n_steps to distribute time reduction evenly in n number of steps
        timesteps = torch.linspace(self.T, 1, n_steps).long()

        for i, t_val in enumerate(timesteps):
            # create a tensor of shape (B,) filled with t_val index to map
            t_int = torch.full((B,), t_val.item(), dtype=torch.long, device=device)

            # considering every token in the sequence using padding mask of 1
            pad_mask = torch.ones(B, L, dtype=torch.bool, device=device)

            # prediction output from the NN output layer
            logits = self.denoiser(xt, t_int, mask=pad_mask, cond=cond)

            # scaling the logits based on temperature
            logits = logits / temparature

            # converting the logits to probabilities using softmax
            probs = F.softmax(logits, dim=-1)

            if top_p < 1.0:
                # finds the top_p probabilities and their indices
                probs = self.filter_top_p(probs, top_p)

            # this exactly samples one from the distribution
            # first it converts the shape of probability to rows only
            # by converted B, L, V to B*L, V
            # applies multinomial sampling to sample 1 element for each row
            # then as the Vocabulary size reduces it reshapes into B, L
            x0_pred = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, L)

            sigma_t = self.scheduler.sigma[t_int[0]]
            # the previous step in self.T is actually the previous step in the scheduler
            # given as t_val - unit (here unit is self.T // n_steps)
            sigma_t_prev = self.scheduler.sigma[
                max(t_val.item() - self.T // n_steps, 0)
            ]

            # when time step not final sigma_t > 0
            if sigma_t > 0:
                # it's basically 1 - (sigma_t_pre / sigma_t)
                unmask_prob = (sigma_t - sigma_t_prev) / sigma_t
            # at final time step unmask everything
            else:
                unmask_prob = 1.0

            masked_pos = xt == self.mask_id
            # create a random distribution of 0 - 1 and get random indices those are less than unmask_prob
            unmask_decision = torch.rand(B, L, device=device) < unmask_prob
            # unmask those indicies as an intersectio of indices those are already masked or random places to unmask in indices
            do_unmask = masked_pos & unmask_decision

            xt = xt.clone()
            # assign predicted values to the unmasked positions
            xt[do_unmask] = x0_pred[do_unmask]

        still_masked = xt == self.mask_id
        if still_masked.any():
            # if there are still masked positions, repeat the process
            # consider the time step to be zero (don't need to call the scheduler)
            t_zero = torch.zeros(B, dtype=torch.long, device=device)
            pad_mask = torch.ones(B, L, dtype=torch.bool, device=device)
            # find logits
            logits = self.denoiser(xt, t_zero, pad_mask, cond=cond)
            # find probs from logits
            probs = F.softmax(logits, dim=-1)
            # use multinomial sampling to get the final predicted tokens
            x0_final = torch.multinomial(probs.view(-1, probs.shape[-1]), 1).view(B, L)
            # replacing the remaining masked positions with the predicted tokens
            xt[still_masked] = x0_final[still_masked]
        still_masked = xt == self.mask_id
        return xt

    @staticmethod
    def filter_top_p(logits, top_p):
        # the top p probabilities setup first sorts the probabilities and keeps their indices (to restore later)
        sorted_probs, sorted_indices = logits.sort(dim=-1, descending=True)
        # finds the cumulative summation
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        # gets the indices of the cumulative probabilities that are greater than top_p
        remove = (cumulative_probs - sorted_probs) > top_p
        # sets the probabilities to zero for the indices that are greater than top_p
        sorted_probs[remove] = 0.0
        # restores the indices to their original order
        probs_filtered = torch.zeros_like(logits).scatter_(
            -1, sorted_indices, sorted_probs
        )
        # normalizes the probabilities ( to again range between 0 and 1)
        probs_filtered /= probs_filtered.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return probs_filtered


if __name__ == "__main__":
    # create an instance of MDPM
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MDPM(
        vocab_size=66,
        pad_id=0,
        mask_id=1,
    ).to(device)

    # create a random input tensor
    x0 = torch.randint(0, 66, (32, 100), device=device)
    print("x0 shape", x0.shape)

    # compute the loss
    loss = model(x0)
    print("loss", loss)

    predict = model.sample(32, 100, device)
    print("preicted tensor shape>>", predict.shape)

    # print the loss
    print(loss)
