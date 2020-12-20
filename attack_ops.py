import numpy as np
from itertools import product, repeat
import PIL
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import transforms
from tv_utils import SpatialAffine, GaussianSmoothing
from attack_utils import projection_linf, check_shape, dlr_loss, get_diff_logits_grads_batch
from imagenet_c import corrupt
import torch.optim as optim
import math
from advertorch.attacks import LinfSPSAAttack
import os

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]

def check_oscillation(x, j, k, y5, k3=0.5):
    t = np.zeros(x.shape[1])
    for counter5 in range(k):
        t += x[j - counter5] > x[j - counter5 - 1]
    return t <= k*k3*np.ones(t.shape)

def CommonCorruptionsAttack(x, y, model, magnitude, name):
    
    x = x.cuda()
    y = y.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, None
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)

    x_np = x.permute((0,2,3,1)).cpu().numpy()  # We make a copy to avoid changing things in-place
    x_np = (x_np * 255).astype(np.uint8)[:,:,::-1]

    for batch_idx, x_np_b in enumerate(x_np):
        corrupt_x = corrupt(x_np_b, corruption_name=name, severity=int(magnitude))
        corrupt_x = corrupt_x.astype(np.float32) / 255.

        adv[ind_non_suc[batch_idx]] = torch.from_numpy(corrupt_x).permute((2,0,1)).cuda()

    return adv, None

def GaussianNoiseAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'gaussian_noise')

def ContrastAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'contrast')

def GaussianBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'gaussian_blur')

def SaturateAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'saturate')

def ElasticTransformAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'elastic_transform')

def JpegCompressionAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'jpeg_compression')

def ShotNoiseAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'shot_noise')

def ImpulseNoiseAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'impulse_noise')

def DefocusBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'defocus_blur')

def GlassBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'glass_blur')

def MotionBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'motion_blur')

def ZoomBlurAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'zoom_blur')

def FogAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'fog')

def BrightnessAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'brightness')

def PixelateAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'pixelate')

def SpeckleNoiseAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'speckle_noise')

def SpatterAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return CommonCorruptionsAttack(x, y, model, magnitude, 'spatter')

def SPSAAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    adversary = LinfSPSAAttack(model, eps=16./255, nb_iter=100)

    x = x.cuda()
    y = y.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, None
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    advimg = adversary.perturb(x, y)

    adv[ind_non_suc] = advimg
    # adv = advimg

    return adv, None

def Skip(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    return x.clone(), previous_p

def DDNL2Attack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='l2', gpu_idx=None):
    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    batch_size = x.shape[0]
    data_dims = (1,) * (x.dim() - 1)
    norm = torch.full((batch_size,), 1, dtype=torch.float).to(device)
    worst_norm = torch.max(x - 0, 1 - x).flatten(1).norm(p=2, dim=1)

    delta = torch.zeros_like(x, requires_grad=True)
    optimizer = torch.optim.SGD([delta], lr=1)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iters, eta_min=0.01)

    best_l2 = worst_norm.clone()
    best_delta = torch.zeros_like(x)

    for i in range(max_iters):
        l2 = delta.data.flatten(1).norm(p=2, dim=1)
        logits = model(x + delta)
        pred_labels = logits.argmax(1)
        
        if target is not None:
            loss = F.cross_entropy(logits, target)
        else:
            loss = -F.cross_entropy(logits, y)

        is_adv = (pred_labels == target) if target is not None else (
            pred_labels != y)
        is_smaller = l2 < best_l2
        is_both = is_adv * is_smaller
        best_l2[is_both] = l2[is_both]
        best_delta[is_both] = delta.data[is_both]

        optimizer.zero_grad()
        loss.backward()

        # renorming gradient
        grad_norms = delta.grad.flatten(1).norm(p=2, dim=1)
        delta.grad.div_(grad_norms.view(-1, *data_dims))
        # avoid nan or inf if gradient is 0
        if (grad_norms == 0).any():
            delta.grad[grad_norms == 0] = torch.randn_like(
                delta.grad[grad_norms == 0])

        optimizer.step()
        scheduler.step()

        norm.mul_(1 - (2 * is_adv.float() - 1) * 0.05)

        delta.data.mul_((norm / delta.data.flatten(1).norm(
            p=2, dim=1)).view(-1, *data_dims))

        delta.data.add_(x)
        delta.data.mul_(255).round_().div_(255)
        delta.data.clamp_(0, 1).sub_(x)
        # print(best_l2)

    adv_imgs = x + best_delta

    dist = (adv_imgs - x)
    dist = dist.view(x.shape[0], -1)
    dist_norm = torch.norm(dist, dim=1, keepdim=True)
    mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
    dist = dist / dist_norm
    dist *= max_eps
    dist = dist.view(x.shape)
    adv_imgs = (x + dist) * mask.float() + adv_imgs * (1 - mask.float())

    if previous_p is not None:
        original_image = x - previous_p
        global_dist = adv_imgs - original_image
        global_dist = global_dist.view(x.shape[0], -1)
        dist_norm = torch.norm(global_dist, dim=1, keepdim=True)
        # print(dist_norm)
        mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
        global_dist = global_dist / dist_norm
        global_dist *= max_eps
        global_dist = global_dist.view(x.shape)
        adv_imgs = (original_image + global_dist) * mask.float() + adv_imgs * (1 - mask.float())
    
    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def CWL2Attack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, kappa=20, _type='linf', gpu_idx=None):

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    delta = torch.zeros_like(x).to(device)
    delta.detach_()
    delta.requires_grad = True

    optimizer = torch.optim.Adam([delta], lr=0.01)
    prev = 1e10

    for step in range(max_iters):

        loss1 = nn.MSELoss(reduction='sum')(delta, torch.zeros_like(x).to(device))

        outputs = model(x+delta)
        one_hot_labels = torch.eye(len(outputs[0]))[y].to(device)

        i, _ = torch.max((1-one_hot_labels)*outputs, dim=1)
        j = torch.masked_select(outputs, one_hot_labels.bool())

        if target is not None:
            one_hot_target_labels = torch.eye(len(outputs[0]))[target].to(device)
            i, _ = torch.max((1-one_hot_target_labels)*outputs, dim=1)
            j = torch.masked_select(outputs, one_hot_target_labels.bool())
            loss2 = torch.sum(torch.clamp(i-j, min=-kappa))
        else:
            loss2 = torch.sum(torch.clamp(j-i, min=-kappa))

        cost = 2*loss1 + loss2

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # print(delta.view(x.size(0), -1).norm(p=2,dim=1))
    adv_imgs = x + delta
    adv_imgs = torch.clamp(adv_imgs, 0, 1)

    dist = (adv_imgs - x)
    dist = dist.view(x.shape[0], -1)
    dist_norm = torch.norm(dist, dim=1, keepdim=True)
    mask = (dist_norm > magnitude).unsqueeze(2).unsqueeze(3)
    dist = dist / dist_norm
    dist *= magnitude
    dist = dist.view(x.shape)
    adv_imgs = (x + dist) * mask.float() + adv_imgs * (1 - mask.float())

    if previous_p is not None:
        original_image = x - previous_p
        global_dist = adv_imgs - original_image
        global_dist = global_dist.view(x.shape[0], -1)
        dist_norm = torch.norm(global_dist, dim=1, keepdim=True)
        mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
        global_dist = global_dist / dist_norm
        global_dist *= max_eps
        global_dist = global_dist.view(x.shape)
        adv_imgs = (original_image + global_dist) * mask.float() + adv_imgs * (1 - mask.float())

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def CWLinfAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    model.eval()
    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    one_hot_y = torch.zeros(y.size(0), 10).to(device)
    one_hot_y[torch.arange(y.size(0)), y] = 1

    # random_start
    x.requires_grad = True 
    if isinstance(magnitude, Variable):
        rand_perturb = torch.FloatTensor(x.shape).uniform_(
                    -magnitude.item(), magnitude.item())
    else:
        rand_perturb = torch.FloatTensor(x.shape).uniform_(
                    -magnitude, magnitude)
    if torch.cuda.is_available():
        rand_perturb = rand_perturb.to(device)
    adv_imgs = x + rand_perturb
    adv_imgs.clamp_(0, 1)

    if previous_p is not None:
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    # max_iters = int(round(magnitude/0.00784) + 2)
    max_iters = int(max_iters)

    with torch.enable_grad():
        for _iter in range(max_iters):
            
            outputs = model(adv_imgs)

            correct_logit = torch.sum(one_hot_y * outputs, dim=1)
            if target is not None:
                wrong_logit = torch.zeros(target.size(0), 10).to(device)
                wrong_logit[torch.arange(target.size(0)), target] = 1
                wrong_logit = torch.sum(wrong_logit * outputs, dim=1)
            else:
                wrong_logit,_ = torch.max((1-one_hot_y) * outputs-1e4*one_hot_y, dim=1)

            loss = -torch.sum(F.relu(correct_logit-wrong_logit+50))

            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]

            adv_imgs.data += 0.00392 * torch.sign(grads.data) 

            # the adversaries' pixel value should within max_x and min_x due 
            # to the l_infinity / l2 restriction

            adv_imgs = torch.max(torch.min(adv_imgs, x + magnitude), x - magnitude)

            adv_imgs.clamp_(0, 1)

            adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)

    adv_imgs.clamp_(0, 1)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def CWLinf_Attack_adaptive_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    # print(x.shape)

    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    one_hot_y = torch.zeros(y.size(0), 10).to(device)
    one_hot_y[torch.arange(y.size(0)), y] = 1
    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        correct_logit = torch.sum(one_hot_y * logits, dim=1)
        wrong_logit,_ = torch.max((1-one_hot_y) * logits-1e4*one_hot_y, dim=1)

        loss_indiv = -F.relu(correct_logit-wrong_logit+50)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            correct_logit = torch.sum(one_hot_y * logits, dim=1)
            wrong_logit,_ = torch.max((1-one_hot_y) * logits-1e4*one_hot_y, dim=1)

            loss_indiv = -F.relu(correct_logit-wrong_logit+50)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv[ind_non_suc] = x_best_adv
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def PGD_Attack_adaptive_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    # print(x.shape)

    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        if target is not None:
            loss_indiv = -F.cross_entropy(logits, target, reduce=False)
        else:
            loss_indiv = F.cross_entropy(logits, y, reduce=False)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            x_adv_old = x_adv.clone()
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            if target is not None:
                loss_indiv = -F.cross_entropy(logits, target, reduce=False)
            else:
                loss_indiv = F.cross_entropy(logits, y, reduce=False)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv[ind_non_suc] = x_best_adv
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def MI_Attack_adaptive_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)

    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps
    else:
        max_x = x + max_eps
        min_x = x - max_eps

    x.requires_grad = True 
 
    n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
    if _type == 'linf':
        t = 2 * torch.rand(x.shape).to(device).detach() - 1
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
        x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
    elif _type == 'l2':
        t = torch.randn(x.shape).to(device).detach()
        x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)

        if previous_p is not None:
            x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
        
    x_adv = x_adv.clamp(0., 1.)
    x_best = x_adv.clone()
    x_best_adv = x_adv.clone()
    loss_steps = torch.zeros([max_iters, x.shape[0]])
    loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
    acc_steps = torch.zeros_like(loss_best_steps)
    
    x_adv.requires_grad_()
    with torch.enable_grad():
        logits = model(x_adv) # 1 forward pass (eot_iter = 1)
        if target is not None:
            loss_indiv = -F.cross_entropy(logits, target, reduce=False)
        else:
            loss_indiv = F.cross_entropy(logits, y, reduce=False)
        loss = loss_indiv.sum()
    grad = torch.autograd.grad(loss, [x_adv])[0].detach()
    
    grad_best = grad.clone()
    acc = logits.detach().max(1)[1] == y
    acc_steps[0] = acc + 0
    loss_best = loss_indiv.detach().clone()

    step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
    x_adv_old = x_adv.clone()

    k = n_iter_2 + 0
    u = np.arange(x.shape[0])
    counter3 = 0
    
    loss_best_last_check = loss_best.clone()
    reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
    n_reduced = 0

    for i in range(max_iters):
        with torch.no_grad():
            x_adv = x_adv.detach()
            grad2 = x_adv - x_adv_old
            x_adv_old = x_adv.clone()
            
            a = 0.75 if i > 0 else 1.0
            
            
            if _type == 'linf':
                x_adv_1 = x_adv + step_size * torch.sign(grad)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - magnitude), x + magnitude), 0.0, 1.0)
                x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                
            elif _type == 'l2':
                x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                if previous_p is not None:
                    x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

            x_adv = x_adv_1 + 0.
        
        x_adv.requires_grad_()

        # print((x_adv-x + previous_p).abs().max())

        with torch.enable_grad():
            logits = model(x_adv) # 1 forward pass (eot_iter = 1)
            if target is not None:
                loss_indiv = -F.cross_entropy(logits, target, reduce=False)
            else:
                loss_indiv = F.cross_entropy(logits, y, reduce=False)
            loss = loss_indiv.sum()
        
        grad = torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
            
        pred = logits.detach().max(1)[1] == y
        acc = torch.min(acc, pred)
        acc_steps[i + 1] = acc + 0
        x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.
        # print((x_best_adv-x + previous_p).abs().max())
        ### check step size
        with torch.no_grad():
            y1 = loss_indiv.detach().clone()
            loss_steps[i] = y1.cpu() + 0
            ind = (y1 > loss_best).nonzero().squeeze()
            x_best[ind] = x_adv[ind].clone()
            grad_best[ind] = grad[ind].clone()
            loss_best[ind] = y1[ind] + 0
            loss_best_steps[i + 1] = loss_best + 0
            
            counter3 += 1
        
            if counter3 == k:
                fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                reduced_last_check = np.copy(fl_oscillation)
                loss_best_last_check = loss_best.clone()
                
                if np.sum(fl_oscillation) > 0:
                    step_size[u[fl_oscillation]] /= 2.0
                    n_reduced = fl_oscillation.astype(float).sum()
                    
                    fl_oscillation = np.where(fl_oscillation)
                    
                    x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                    grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                    
                counter3 = 0
                k = np.maximum(k - size_decr, n_iter_min)


    adv[ind_non_suc] = x_best_adv
    now_p = x_best_adv-x
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def ODI_Cos_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    def margin_loss(logits,y):
        logit_org = logits.gather(1,y.view(-1,1))
        logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
        loss = -logit_org + logit_target
        loss = torch.sum(loss)
        return loss

    def cos_lr(iteration, max_iteration):
        iteration = iteration % max_iteration
        lr = 0.00031 + (0.031-0.00031) * (1 + math.cos(math.pi * iteration / max_iteration)) / 2
        return lr

    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    x.requires_grad = True 
    
    randVector_ = torch.FloatTensor(*model(x).shape).uniform_(-1.,1.).to(device)

    rand_perturb = torch.FloatTensor(x.shape).uniform_(
                -magnitude, magnitude)
    if torch.cuda.is_available():
        rand_perturb = rand_perturb.to(device)
    adv_imgs = x + rand_perturb
    adv_imgs.clamp_(0, 1)

    # max_iters = int(round(magnitude/0.00784) + 2)
    max_iters = int(max_iters)

    for i in range(10 + max_iters):
        with torch.enable_grad():
            if i < 10:
                loss = (model(adv_imgs) * randVector_).sum()
            else:
                loss = margin_loss(model(adv_imgs),y)

        grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]
        if i < 10: 
            eta = 0.031 * grads.data.sign()
        elif i>=10:
            eta = cos_lr(i-10, max_iters) * grads.data.sign()
        
        adv_imgs = Variable(adv_imgs.data + eta, requires_grad=True)

        if previous_p is not None:
            max_x = x - previous_p + max_eps
            min_x = x - previous_p - max_eps
        else:
            max_x = x + max_eps
            min_x = x - max_eps
        adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)
        adv_imgs = Variable(torch.clamp(adv_imgs, 0, 1.0), requires_grad=True)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def ODI_Cyclical_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    def margin_loss(logits,y):
        logit_org = logits.gather(1,y.view(-1,1))
        logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
        loss = -logit_org + logit_target
        loss = torch.sum(loss)
        return loss

    def cycle_step(iteration, max_iteration):
        cycle = np.floor(1+iteration/(max_iteration/3))
        x = np.abs(iteration/(max_iteration/6) - 2*cycle + 1)
        lr = 0.00031 + (0.031-0.00031)*np.maximum(0, (1-x))
        return lr

    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    x.requires_grad = True 
    
    randVector_ = torch.FloatTensor(*model(x).shape).uniform_(-1.,1.).to(device)

    rand_perturb = torch.FloatTensor(x.shape).uniform_(
                -magnitude, magnitude)
    if torch.cuda.is_available():
        rand_perturb = rand_perturb.to(device)
    adv_imgs = x + rand_perturb
    adv_imgs.clamp_(0, 1)

    # max_iters = int(round(magnitude/0.00784) + 2)
    max_iters = int(max_iters)

    for i in range(10 + max_iters):
        with torch.enable_grad():
            if i < 10:
                loss = (model(adv_imgs) * randVector_).sum()
            else:
                loss = margin_loss(model(adv_imgs),y)

        grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]
        if i < 10: 
            eta = 0.031 * grads.data.sign()
        elif i>=10:
            eta = cycle_step(i-10, max_iters) * grads.data.sign()
        
        adv_imgs = Variable(adv_imgs.data + eta, requires_grad=True)

        if previous_p is not None:
            max_x = x - previous_p + max_eps
            min_x = x - previous_p - max_eps
        else:
            max_x = x + max_eps
            min_x = x - max_eps
        adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)
        adv_imgs = Variable(torch.clamp(adv_imgs, 0, 1.0), requires_grad=True)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p



def ODI_Step_stepsize(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):
    
    def margin_loss(logits,y):
        logit_org = logits.gather(1,y.view(-1,1))
        logit_target = logits.gather(1,(logits - torch.eye(10)[y].to("cuda") * 9999).argmax(1, keepdim=True))
        loss = -logit_org + logit_target
        loss = torch.sum(loss)
        return loss

    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    x.requires_grad = True 
    
    randVector_ = torch.FloatTensor(*model(x).shape).uniform_(-1.,1.).to(device)

    rand_perturb = torch.FloatTensor(x.shape).uniform_(
                -magnitude, magnitude)
    if torch.cuda.is_available():
        rand_perturb = rand_perturb.to(device)
    adv_imgs = x + rand_perturb
    adv_imgs.clamp_(0, 1)

    # max_iters = int(round(magnitude/0.00784) + 2)
    max_iters = int(max_iters)

    for i in range(10 + max_iters):
        with torch.enable_grad():
            if i < 10:
                loss = (model(adv_imgs) * randVector_).sum()
            else:
                loss = margin_loss(model(adv_imgs),y)

        grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]
        if i < 10: 
            eta = 0.031 * grads.data.sign()
        elif i>=10 and i<60:
            eta = 0.031 * grads.data.sign()
        elif i>=60 and i<110:
            eta = 0.0031 * grads.data.sign()
        elif i>=110:
            eta = 0.00031 * grads.data.sign()
        
        adv_imgs = Variable(adv_imgs.data + eta, requires_grad=True)

        if previous_p is not None:
            max_x = x - previous_p + max_eps
            min_x = x - previous_p - max_eps
        else:
            max_x = x + max_eps
            min_x = x - max_eps
        adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)
        adv_imgs = Variable(torch.clamp(adv_imgs, 0, 1.0), requires_grad=True)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p
        
def SpatialAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', limits_factor=[5, 5, 31], granularity=[5, 5, 5], gpu_idx=None):
    
    model.eval()

    x = x.cuda()
    y = y.cuda()
    if target is not None:
        target = target.cuda()
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
 
    n = x.size(0)
    limits = [x for x in limits_factor]

    grid = product(*list(np.linspace(-l, l, num=g) for l, g in zip(limits, granularity)))

    worst_x = x.clone()
    worst_t = torch.zeros([n, 3]).cuda()
    max_xent = -torch.ones(n).cuda() * 1e8
    all_correct = torch.ones(n).cuda().bool()

    for tx, ty, r in grid:
        spatial_transform = transforms.Compose([
            SpatialAffine(degrees=r, translate=(tx, ty), resample=PIL.Image.BILINEAR),
            transforms.ToTensor()
    ])    

        img_list = []
        for i in range(x.shape[0]):
            x_pil = transforms.ToPILImage()(x[i,:,:,:].cpu())
            adv_img_tensor = spatial_transform(x_pil)
            img_list.append(adv_img_tensor)
        adv_input = torch.stack(img_list).cuda()
        with torch.no_grad():
            output = model(adv_input)
        # output = self.model(torch.from_numpy(x_nat).to(device)).cpu()

        cur_xent = F.cross_entropy(output, y, reduce=False)
        cur_correct = output.max(1)[1]==y

        # of maximum xent (or just highest xent if everything else if correct).
        idx = (cur_xent > max_xent) & (cur_correct == all_correct)
        idx = idx | (cur_correct < all_correct)
        max_xent = torch.where(cur_xent>max_xent, cur_xent, max_xent)
        # max_xent = np.maximum(cur_xent, max_xent)
        all_correct = cur_correct & all_correct

        idx = idx.unsqueeze(-1) # shape (bsize, 1)
        worst_t = torch.where(idx, torch.from_numpy(np.array([tx, ty, r]).astype(np.float32)).cuda(), worst_t) # shape (bsize, 3)
        idx = idx.unsqueeze(-1)
        idx = idx.unsqueeze(-1) # shape (bsize, 1, 1, 1)
        worst_x = torch.where(idx, adv_input, worst_x) # shape (bsize, 32, 32, 3)


    adv[ind_non_suc] = worst_x
    # adv = worst_x

    return adv, None

def MultiTargetedAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, target=None, _type='linf', gpu_idx=None):

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv_out = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)
    
    def run_once(model, x_in, y_in, magnitude, max_iters, _type, target_class, max_eps, previous_p):
        x = x_in.clone() if len(x_in.shape) == 4 else x_in.clone().unsqueeze(0)
        y = y_in.clone() if len(y_in.shape) == 1 else y_in.clone().unsqueeze(0)
        # print(x.shape)
        if previous_p is not None:
            max_x = x - previous_p + max_eps
            min_x = x - previous_p - max_eps
        else:
            max_x = x + max_eps
            min_x = x - max_eps

        n_iter_2, n_iter_min, size_decr = max(int(0.22 * max_iters), 1), max(int(0.06 * max_iters), 1), max(int(0.03 * max_iters), 1)
        if _type == 'linf':
            t = 2 * torch.rand(x.shape).to(device).detach() - 1
            x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / (t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))
            x_adv = torch.clamp(torch.min(torch.max(x_adv, min_x), max_x), 0.0, 1.0)
        elif _type == 'l2':
            t = torch.randn(x.shape).to(device).detach()
            x_adv = x.detach() + magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * t / ((t ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
            if previous_p is not None:
                x_adv = torch.clamp(x - previous_p + (x_adv - x + previous_p) / (((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                    max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

        x_adv = x_adv.clamp(0., 1.)
        x_best = x_adv.clone()
        x_best_adv = x_adv.clone()
        loss_steps = torch.zeros([max_iters, x.shape[0]])
        loss_best_steps = torch.zeros([max_iters + 1, x.shape[0]])
        acc_steps = torch.zeros_like(loss_best_steps)

        output = model(x)
        y_target = output.sort(dim=1)[1][:, -target_class]
        
        x_adv.requires_grad_()
        grad = torch.zeros_like(x)
        for _ in range(1):
            with torch.enable_grad():
                logits = model(x_adv) # 1 forward pass (eot_iter = 1)
                loss_indiv = dlr_loss(logits, y, y_target)
                loss = loss_indiv.sum()
            grad += torch.autograd.grad(loss, [x_adv])[0].detach()
        
        grad_best = grad.clone()
        acc = logits.detach().max(1)[1] == y
        acc_steps[0] = acc + 0
        loss_best = loss_indiv.detach().clone()

        step_size = magnitude * torch.ones([x.shape[0], 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])
        x_adv_old = x_adv.clone()
        counter = 0
        k = n_iter_2 + 0
        u = np.arange(x.shape[0])
        counter3 = 0
        
        loss_best_last_check = loss_best.clone()
        reduced_last_check = np.zeros(loss_best.shape) == np.zeros(loss_best.shape)
        n_reduced = 0

        for i in range(max_iters):
            with torch.no_grad():
                x_adv = x_adv.detach()
                grad2 = x_adv - x_adv_old
                x_adv_old = x_adv.clone()
                
                a = 0.75 if i > 0 else 1.0
                
                
                if _type == 'linf':
                    x_adv_1 = x_adv + step_size * torch.sign(grad)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, x - magnitude), x + magnitude), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a), x - magnitude), x + magnitude), 0.0, 1.0)
                    x_adv_1 = torch.clamp(torch.min(torch.max(x_adv_1, min_x), max_x), 0.0, 1.0)
                    
                elif _type == 'l2':
                    x_adv_1 = x_adv + step_size[0] * grad / ((grad ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt()), 0.0, 1.0)
                    x_adv_1 = x_adv + (x_adv_1 - x_adv)*a + grad2*(1 - a)
                    x_adv_1 = torch.clamp(x + (x_adv_1 - x) / (((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                        magnitude * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)

                    if previous_p is not None:
                        x_adv_1 = torch.clamp(x - previous_p + (x_adv_1 - x + previous_p) / (((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12) * torch.min(
                            max_eps * torch.ones(x.shape).to(device).detach(), ((x_adv_1 - x + previous_p) ** 2).sum(dim=(1, 2, 3), keepdim=True).sqrt() + 1e-12), 0.0, 1.0)
                   
                x_adv = x_adv_1 + 0.
            
            x_adv.requires_grad_()
            grad = torch.zeros_like(x)
            for _ in range(1):
                with torch.enable_grad():
                    logits = model(x_adv) # 1 forward pass (eot_iter = 1)
                    loss_indiv = dlr_loss(logits, y, y_target)
                    loss = loss_indiv.sum()
                
                grad += torch.autograd.grad(loss, [x_adv])[0].detach() # 1 backward pass (eot_iter = 1)
                
            pred = logits.detach().max(1)[1] == y
            acc = torch.min(acc, pred)
            acc_steps[i + 1] = acc + 0
            x_best_adv[(pred == 0).nonzero().squeeze()] = x_adv[(pred == 0).nonzero().squeeze()] + 0.

            ### check step size
            with torch.no_grad():
              y1 = loss_indiv.detach().clone()
              loss_steps[i] = y1.cpu() + 0
              ind = (y1 > loss_best).nonzero().squeeze()
              x_best[ind] = x_adv[ind].clone()
              grad_best[ind] = grad[ind].clone()
              loss_best[ind] = y1[ind] + 0
              loss_best_steps[i + 1] = loss_best + 0
              
              counter3 += 1
          
              if counter3 == k:
                  fl_oscillation = check_oscillation(loss_steps.detach().cpu().numpy(), i, k, loss_best.detach().cpu().numpy(), k3=.75)
                  fl_reduce_no_impr = (~reduced_last_check) * (loss_best_last_check.cpu().numpy() >= loss_best.cpu().numpy())
                  fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                  reduced_last_check = np.copy(fl_oscillation)
                  loss_best_last_check = loss_best.clone()
                  
                  if np.sum(fl_oscillation) > 0:
                      step_size[u[fl_oscillation]] /= 2.0
                      n_reduced = fl_oscillation.astype(float).sum()
                      
                      fl_oscillation = np.where(fl_oscillation)
                      
                      x_adv[fl_oscillation] = x_best[fl_oscillation].clone()
                      grad[fl_oscillation] = grad_best[fl_oscillation].clone()
                      
                  counter3 = 0
                  k = np.maximum(k - size_decr, n_iter_min)

        return acc, x_best_adv


    adv = x.clone()
    for target_class in range(2, 9 + 2):
        acc_curr, adv_curr = run_once(model, x, y, magnitude, max_iters, _type, target_class, max_eps, previous_p)
        ind_curr = (acc_curr == 0).nonzero().squeeze()
        adv[ind_curr] = adv_curr[ind_curr].clone()

    now_p = adv-x
    adv_out[ind_non_suc] = adv
    # print(adv_out==x)
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv_out, previous_p_c

    return adv_out, now_p

def MomentumIterativeAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=20, decay_factor=1., target=None, _type='linf', gpu_idx=None):

    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    adv_imgs = x
    adv_imgs.requires_grad = True 
    adv_imgs = torch.clamp(adv_imgs, min=0, max=1)

    # max_iters = 20
    max_iters = int(max_iters)
    with torch.enable_grad():
        for i in range(max_iters):
            outputs = model(adv_imgs)

            if target is not None:
                loss = -F.cross_entropy(outputs, target)
            else:
                loss = F.cross_entropy(outputs, y)

            grads = torch.autograd.grad(loss, adv_imgs, grad_outputs=None, 
                    only_inputs=True)[0]

            grad_norm = grads.data.abs().pow(1).view(adv_imgs.size(0), -1).sum(dim=1).pow(1)

            grad_norm = torch.max(grad_norm, torch.ones_like(grad_norm) * 1e-6)

            g = (grads.data.transpose(0, -1) * grad_norm).transpose(0, -1).contiguous()

            g = decay_factor * g + (grads.data.transpose(0, -1) * grad_norm).transpose(0, -1).contiguous()

            adv_imgs.data += 0.00392 * torch.sign(g)

            if _type == 'linf':
                if previous_p is not None:
                    max_x = x - previous_p + max_eps
                    min_x = x - previous_p - max_eps

                else:
                    max_x = x + max_eps
                    min_x = x - max_eps
                adv_imgs = torch.max(torch.min(adv_imgs, x + magnitude), x - magnitude)
                adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)
            elif _type == 'l2':
                dist = (adv_imgs - x)
                dist = dist.view(x.shape[0], -1)
                dist_norm = torch.norm(dist, dim=1, keepdim=True)
                mask = (dist_norm > magnitude).unsqueeze(2).unsqueeze(3)
                dist = dist / dist_norm
                dist *= magnitude
                dist = dist.view(x.shape)
                adv_imgs = (x + dist) * mask.float() + x * (1 - mask.float())

                if previous_p is not None:
                    original_image = x - previous_p
                    global_dist = adv_imgs - original_image
                    global_dist = global_dist.view(x.shape[0], -1)
                    dist_norm = torch.norm(global_dist, dim=1, keepdim=True)
                    mask = (dist_norm > max_eps).unsqueeze(2).unsqueeze(3)
                    global_dist = global_dist / dist_norm
                    global_dist *= max_eps
                    global_dist = global_dist.view(x.shape)
                    adv_imgs = (original_image + global_dist) * mask.float() + adv_imgs * (1 - mask.float())

            adv_imgs.clamp_(0, 1)

    adv_imgs.clamp_(0, 1)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def GradientSignAttack(x, y, model, magnitude, previous_p, max_eps, max_iters=1, target=None, _type='linf', gpu_idx=None):
    model.eval()

    device = 'cuda:{}'.format(gpu_idx)
    x = x.to(device)
    y = y.to(device)
    if target is not None:
        target = target.to(device)
    adv = x.clone()
    pred = predict_from_logits(model(x))
    if torch.sum((pred==y)).item() == 0:
        return adv, previous_p
    ind_non_suc = (pred==y).nonzero().squeeze()
    x = x[ind_non_suc]
    y = y[ind_non_suc]
    x = x if len(x.shape) == 4 else x.unsqueeze(0)
    y = y if len(y.shape) == 1 else y.unsqueeze(0)
    if previous_p is not None:
        previous_p = previous_p.to(device)
        previous_p_c = previous_p.clone()
        previous_p = previous_p[ind_non_suc]
        previous_p = previous_p if len(previous_p.shape) == 4 else previous_p.unsqueeze(0)

    adv_imgs = x
    adv_imgs.requires_grad = True 

    # in FGSM attack, max_iters must be 1
    assert max_iters == 1

    if previous_p is not None:
        max_x = x - previous_p + max_eps
        min_x = x - previous_p - max_eps

    else:
        max_x = x + max_eps
        min_x = x - max_eps

    outputs = model(adv_imgs)

    if target is not None:
        loss = -F.cross_entropy(outputs, target)
    else:
        loss = F.cross_entropy(outputs, y)

    loss.backward()
    grad_sign = adv_imgs.grad.sign()

    pertubation = magnitude * grad_sign
    adv_imgs = torch.clamp(adv_imgs + pertubation,0,1)
    adv_imgs = torch.max(torch.min(adv_imgs, max_x), min_x)

    now_p = adv_imgs-x
    adv[ind_non_suc] = adv_imgs
    if previous_p is not None:
        previous_p_c[ind_non_suc] = previous_p + now_p
        return adv, previous_p_c

    return adv, now_p

def attacker_list():  # 16 operations and their ranges
    l = [GradientSignAttack, 
         PGD_Attack_adaptive_stepsize,
         MI_Attack_adaptive_stepsize,
         CWLinf_Attack_adaptive_stepsize,
         ODI_Cos_stepsize,
         ODI_Cyclical_stepsize,
         ODI_Step_stepsize,
         MultiTargetedAttack,
         CWL2Attack,
         DDNL2Attack,
         Skip, 
         GaussianBlurAttack,
         GaussianNoiseAttack,
         ContrastAttack,
         SaturateAttack,
         ElasticTransformAttack,
         JpegCompressionAttack,
         ShotNoiseAttack,
         ImpulseNoiseAttack,
         DefocusBlurAttack,
         GlassBlurAttack,
         MotionBlurAttack,
         ZoomBlurAttack,
         FogAttack,
         BrightnessAttack,
         PixelateAttack,
         SpeckleNoiseAttack,
         SpatterAttack,
         SPSAAttack,
         SpatialAttack
    ]
    return l


attacker_dict = {fn.__name__: fn for fn in attacker_list()}

def get_attacker(name):
    return attacker_dict[name]

def apply_attacker(img, name, y, model, magnitude, p, steps, max_eps, target=None, _type=None, gpu_idx=None):
    augment_fn = get_attacker(name)
    return augment_fn(x=img, y=y, model=model, magnitude=magnitude, previous_p=p, max_iters=steps,max_eps=max_eps, target=target, _type=_type, gpu_idx=gpu_idx)