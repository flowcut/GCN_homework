import numpy as np
import torch


class LapReg(torch.nn.Module):
    def __init__(self, data):
        super(LapReg, self).__init__()
        dim = data.y.shape[0]
        self.adjmat = np.zeros((dim, dim))
        self.adjmat[data.edge_index[0, :], data.edge_index[1, :]] = 1
        self.adjmat += (self.adjmat.transpose() > self.adjmat) + np.identity(dim)
        # undirected
        d = np.diag(np.sum(self.adjmat, 1))
        self.delta = torch.FloatTensor(d - self.adjmat)

    def to(self, device):
        self.device = device
        super().to(device)
        return self

    def reg_loss(self, x):
        return torch.sum(torch.mm(x.permute(1, 0).contiguous(), torch.mm(self.delta, x)))
