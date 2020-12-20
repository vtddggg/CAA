import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
import numpy as np

device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

class NormalizeByChannelMeanStd(nn.Module):
    def __init__(self, mean, std):
        super(NormalizeByChannelMeanStd, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)

    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)

def get_diff_logits_grads_batch(model, imgs, la):
    im = imgs.clone().requires_grad_()
    with torch.enable_grad():
        y = model(im)

    g2 = torch.zeros([y.shape[-1], *imgs.size()]).to(device)
    grad_mask = torch.zeros_like(y)
    for counter in range(y.shape[-1]):
        zero_gradients(im)
        grad_mask[:, counter] = 1.0
        y.backward(grad_mask, retain_graph=True)
        grad_mask[:, counter] = 0.0
        g2[counter] = im.grad.data

    g2 = torch.transpose(g2, 0, 1).detach()
    y2 = model(imgs).detach()
    df = y2 - y2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
    dg = g2 - g2[torch.arange(imgs.shape[0]), la].unsqueeze(1)
    df[torch.arange(imgs.shape[0]), la] = 1e10

    return df, dg

def check_shape(x):
    return x if len(x.shape) > 0 else x.unsqueeze(0)

def dlr_loss(x, y, y_target):
    x_sorted, ind_sorted = x.sort(dim=1)
    return -(x[np.arange(x.shape[0]), y] - x[np.arange(x.shape[0]), y_target]) / (x_sorted[:, -1] - .5 * x_sorted[:, -3] - .5 * x_sorted[:, -4] + 1e-12)


def projection_linf(points_to_project, w_hyperplane, b_hyperplane):
    t = points_to_project.clone()
    w = w_hyperplane.clone()
    b = b_hyperplane.clone()

    ind2 = ((w * t).sum(1) - b < 0).nonzero().squeeze()
    ind2 = check_shape(ind2)
    w[ind2] *= -1
    b[ind2] *= -1

    c5 = (w < 0).float()
    a = torch.ones(t.shape).to(device)
    d = (a * c5 - t) * (w != 0).float()
    a -= a * (1 - c5)

    p = torch.ones(t.shape).to(device) * c5 - t * (2 * c5 - 1)
    _, indp = torch.sort(p, dim=1)

    b = b - (w * t).sum(1)
    b0 = (w * d).sum(1)
    b1 = b0.clone()

    counter = 0
    indp2 = torch.flip(indp.unsqueeze(-1), dims=(1, 2)).squeeze()
    u = torch.arange(0, w.shape[0])
    ws = w[u.unsqueeze(1), indp2]
    bs2 = - ws * d[u.unsqueeze(1), indp2]

    s = torch.cumsum(ws.abs(), dim=1)
    sb = torch.cumsum(bs2, dim=1) + b0.unsqueeze(1)

    c = b - b1 > 0
    b2 = sb[u, -1] - s[u, -1] * p[u, indp[u, 0]]
    c_l = (b - b2 > 0).nonzero().squeeze()
    c2 = ((b - b1 > 0) * (b - b2 <= 0)).nonzero().squeeze()
    c_l = check_shape(c_l)
    c2 = check_shape(c2)

    lb = torch.zeros(c2.shape[0])
    ub = torch.ones(c2.shape[0]) * (w.shape[1] - 1)
    nitermax = torch.ceil(torch.log2(torch.tensor(w.shape[1]).float()))
    counter2 = torch.zeros(lb.shape).long()

    while counter < nitermax:
        counter4 = torch.floor((lb + ub) / 2)
        counter2 = counter4.long()
        indcurr = indp[c2, -counter2 - 1]
        b2 = sb[c2, counter2] - s[c2, counter2] * p[c2, indcurr]
        c = b[c2] - b2 > 0
        ind3 = c.nonzero().squeeze()
        ind32 = (~c).nonzero().squeeze()
        ind3 = check_shape(ind3)
        ind32 = check_shape(ind32)
        lb[ind3] = counter4[ind3]
        ub[ind32] = counter4[ind32]
        counter += 1

    lb = lb.long()
    counter2 = 0

    if c_l.nelement() != 0:
        lmbd_opt = (torch.max((b[c_l] - sb[c_l, -1]) / (-s[c_l, -1]),
                                torch.zeros(sb[c_l, -1].shape)
                                .to(device))).unsqueeze(-1)
        d[c_l] = (2 * a[c_l] - 1) * lmbd_opt

    lmbd_opt = (torch.max((b[c2] - sb[c2, lb]) / (-s[c2, lb]),
                            torch.zeros(sb[c2, lb].shape)
                            .to(device))).unsqueeze(-1)
    d[c2] = torch.min(lmbd_opt, d[c2]) * c5[c2]\
        + torch.max(-lmbd_opt, d[c2]) * (1 - c5[c2])

    return d * (w != 0).float()