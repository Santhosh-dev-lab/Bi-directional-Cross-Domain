import torch
import torch.nn as nn


class MMD_loss(nn.Module):
    """
    Maximum Mean Discrepancy (MMD) with Gaussian (RBF) kernel.
    Calculates marginal distribution discrepancy in Hilbert space (RKHS)
    between source and target feature representations (Eq. 13 & 14 in paper).
    """
    def __init__(self, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = fix_sigma

    def gaussian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size(0)) + int(target.size(0))
        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
        total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
        l2_distance = ((total0 - total1) ** 2).sum(2)

        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(l2_distance.data) / max(n_samples ** 2 - n_samples, 1)
        
        # Guard against zero or NaN bandwidth
        bandwidth = torch.clamp(bandwidth, min=1e-5)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]

        kernel_val = [torch.exp(-l2_distance / bw) for bw in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        """
        Computes MMD between source and target feature representations.
        Accepts 2D tensors (B, D) or 3D token representations (B, L, C)
        which are automatically flattened per sample to (B, L*C).
        """
        if source.dim() > 2:
            source = source.flatten(1)
        if target.dim() > 2:
            target = target.flatten(1)

        n_s = source.size(0)
        n_t = target.size(0)

        kernels = self.gaussian_kernel(source, target,
                                       kernel_mul=self.kernel_mul,
                                       kernel_num=self.kernel_num,
                                       fix_sigma=self.fix_sigma)
        XX = kernels[:n_s, :n_s]
        YY = kernels[n_s:, n_s:]
        XY = kernels[:n_s, n_s:]
        YX = kernels[n_s:, :n_s]

        # Handles both equal and unequal batch sizes cleanly
        loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
        return torch.clamp(loss, min=0.0)
