import math

import torch


class MaskedDiffusionSchedule:
    def __init__(self, T: int = 1000):
        """ """
        self.T = T  # this is the number of steps in the diffusion process
        ts = torch.linspace(
            0, 1, T + 1
        )  # this is the distribution of from 0 to 1 for T number of points
        # setting up cosine schedule
        self.sigma = (
            1 - torch.cos(ts * math.pi / 2)
        )  # this computes the cosine of the `ts` as distribution (this increases from 0 to 1)

    def get_sigma(self, t):
        """ """
        return self.sigma[
            t
        ]  # returning the sigma value (diffusion coefficient) at step t (time)

    def q_sample(self, x0, t_int, mask_id):
        """
        The sampling setup basically takes the step value to get the sigma (which regulates noise)
        and then applies the noise to the input tensor based on the mask probability and sigma.

        if t_int is to the near complete step (T) there is not noise
        if t_int is near the start step (0) there is maximum noise
        """
        self.sigma = self.sigma.to(x0.device)
        # the shape of x0 is (B, L) and sigma is of shape (B,)
        # hence unsqueeze here allows to implement shape (B, 1) for broadcasting
        sigma = self.get_sigma(t_int).unsqueeze(
            1
        )  # get the sigma value for time step t
        mask_prob = torch.rand_like(
            x0.float()
        )  # random distribute the probability for input tensor
        noise_ids = (
            mask_prob < sigma
        )  # filter the noise ids based on the mask probability and sigma
        xt = x0.clone()  # clone the input tensor
        xt[noise_ids] = mask_id  # assigne the mask id to noise specific tokens
        return xt

    def get_loss_weight(self, t_int):
        sigma_t = self.sigma[t_int]  # sigma value at time step t
        sigma_t1 = self.sigma[
            (t_int + 1).clamp(max=self.T)
        ]  # sigma value at time step t+1
        dsigma = (sigma_t1 - sigma_t) / (
            1.0 / self.T
        )  # the partial derivative of sigma with respect to time (approximated)
        weight = dsigma / (
            sigma_t * (1 - sigma_t) + 1e-8
        )  # calculate the loss weight based on the partial derivative of sigma
        return weight.clamp(0, 100)  # clamp the weight to a reasonable range


if __name__ == "__main__":
    schedule = MaskedDiffusionSchedule(10)
    schedule.q_sample(torch.randn([10, 32]), torch.arange(10), 5)
    loss = schedule.get_loss_weight(torch.tensor(9))
    print(f"{loss:.4f}")
