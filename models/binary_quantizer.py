import torch.nn as nn
import numpy as np
import torch


class SpikeFunction(nn.Module):
    def __init__(self, dampening_factor=0.3):
        super(SpikeFunction, self).__init__()
        self.dampening_factor = dampening_factor

    def forward(self, x, hidden=None):
        z_forward = torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        if not x.requires_grad:
            return z_forward

        z_backward = torch.where(x > 0, - 0.5 * torch.square(x - 1), 0.5 * torch.square(x + 1))
        z_backward = torch.where(torch.abs(x) < 1, z_backward, torch.zeros_like(x))
        z_backward = self.dampening_factor * z_backward

        z = (z_forward - z_backward).detach() + z_backward
        return z


class BinaryQuantizer(nn.Module):

    def __init__(self, n, m, mom=0.95,  lr_internal=0.3, post_normalize=False):
        super(BinaryQuantizer, self).__init__()

        self.n = n
        self.m = m
        self.mom = mom
        self.post_normalize = post_normalize
        self.lr_internal = lr_internal

        self.spike_function = SpikeFunction()

        self.register_buffer("x0", torch.zeros(n))
        self.register_buffer("V", torch.zeros([m,n]))

        self.W = nn.Parameter(data=torch.randn([n,m]) / np.sqrt(n + m), requires_grad=True) # encoding weights

    def forward(self, x):
        shp = x.shape
        assert shp[-1] == self.n
        x = x.reshape(-1, self.n) #

        mom = self.mom

        if self.training:
            # batch norm style normalization
            self.x0.data = mom * self.x0 + (1-mom) * x.mean(0) # assumes dimension [batch x dims]

        x = x - self.x0

        # binarize
        a = x @ self.W
        z_binary = a > 0
        z = self.spike_function(a)

        if self.training and (z.sum() > 0):
            grad = 2 * torch.einsum("bi,bj->ij", z, z @ self.V - x) / x.numel()
            V0 = self.V - self.lr_internal * grad
            self.V.data = self.mom * self.V + (1-self.mom) * V0.detach()

        V0 = self.V

        x_out = z @ V0

        x_out = x_out + self.x0

        if self.post_normalize:
            x_out = normalize(x_out)

        z_shp = [s for s in shp[:-1]] + [self.m]
        x_out = x_out.reshape(shp)
        z_binary = z_binary.reshape(z_shp)

        return x_out, z_binary, z

def normalize(x):
    x = x / torch.norm(x, dim=1, keepdim=True)
    return x

def relative_diff(t1,t2):
    return (t1 - t2).abs().sum() / t1.abs().sum()


if __name__ == '__main__':

    n = 64
    m = 4096
    time_dim = 256
    batch_size = 32

    q = BinaryQuantizer(n,m)
    q.train()

    for _ in range(1000):
        x = torch.randn([batch_size,time_dim,n])
        x_out, code, loss = q(x)
        print(f"err: {relative_diff(x, x_out).item():0.3f}")