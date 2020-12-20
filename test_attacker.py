import sys
import torch
import torch.nn as nn
from torch.nn import functional as F
from PIL import Image
import numpy as np
import os
import torchvision
import imageio

from torchvision import transforms
import argparse

from attack_ops import apply_attacker
from tqdm import tqdm
import bird_or_bicycle
from tv_utils import ImageNet,Permute

import random

# copy from advertorch
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

def normalize_fn(tensor, mean, std):
    """Differentiable version of torchvision.functional.normalize"""
    # here we assume the color channel is in at dim=1
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)

def predict_from_logits(logits, dim=1):
    return logits.max(dim=dim, keepdim=False)[1]


parser = argparse.ArgumentParser(description='Random search of Auto-attack')

parser.add_argument('--seed', type=int, default=2020, help='random seed')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size for data loader')
parser.add_argument('--dataset', default='cifar10', help='cifar10 | cifar100 | svhn | ile')
parser.add_argument('--num_classes', type=int, default=10, help='the # of classes')
parser.add_argument('--net_type', default='madry_adv_resnet50', help='resnet18 | resnet50 | inception_v3 | densenet121 | vgg16_bn')
parser.add_argument('--num_restarts', type=int, default=1, help='the # of classes')
parser.add_argument('--max_epsilon', type=float, default=8/255, help='the attack sequence length')
parser.add_argument('--ensemble', action='store_true', help='the attack sequence length')
parser.add_argument('--transfer_test', action='store_true', help='the attack sequence length')
parser.add_argument('--sub_net_type', default='madry_adv_resnet50', help='resnet18 | resnet50 | inception_v3 | densenet121 | vgg16_bn')
parser.add_argument('--target', action='store_true', default=False)
parser.add_argument('--norm', default='linf', help='linf | l2 | unrestricted')


args = parser.parse_args()
print(args)

# Linf attack policy searched by CAA
subpolicy_linf = [{'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 50}, {'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 25}, {'attacker': 'CWLinf_Attack_adaptive_stepsize', 'magnitude': 8/255, 'step': 125}]

# L2 attack policy searched by CAA
subpolicy_l2 = [{'attacker': 'MultiTargetedAttack', 'magnitude': 0.5, 'step': 100}, {'attacker': 'PGD_Attack_adaptive_stepsize', 'magnitude': 0.4375, 'step': 25}, {'attacker': 'DDNL2Attack', 'magnitude': None, 'step': 1000}]

# unrestricted attack policy searched by CAA
subpolicy_un = [{'attacker': 'FogAttack', 'magnitude': 1.0, 'step': 1}, {'attacker': 'FogAttack', 'magnitude': 1.0, 'step': 1}, {'attacker': 'SPSAAttack', 'magnitude': 1.0, 'step': 1}]

if args.norm == 'linf':
    subpolicy = subpolicy_linf
elif args.norm == 'l2':
    subpolicy = subpolicy_l2
elif args.norm == 'unrestricted':
    subpolicy = subpolicy_un

print('test policy: ', subpolicy, '...')
## define model
device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')

# print(subpolices)
## obtain dataset
if args.dataset == 'mnist':
    args.num_classes = 10
    mnist_val = torchvision.datasets.MNIST(root='/root/project/data/mnist', train=False, transform = transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(mnist_val, batch_size=args.batch_size,
                                                shuffle=False, pin_memory=True, num_workers=8)
    if args.net_type == 'TRADES':
        from mnist_models.small_cnn import SmallCNN
        model = SmallCNN()
        model.load_state_dict(torch.load('./checkpoints/model_mnist_smallcnn.pt'))

    
if args.dataset == 'cifar10':
    args.num_classes = 10
    cifar10_val = torchvision.datasets.CIFAR10(root='/root/project/data/cifar10', train=False, transform = transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(cifar10_val, batch_size=args.batch_size,
                                                shuffle=False, pin_memory=True, num_workers=8)
    if args.net_type == 'clean':
        from cifar_models.resnet import resnet50
        model = resnet50()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_nat.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model)

    elif args.net_type == 'TRADES':
        from cifar_models.wideresnet import WideResNet
        model = WideResNet()
        model.load_state_dict(torch.load('./checkpoints/model-wideres-epoch76.pt'))

    elif args.net_type == 'unlabel':
        from cifar_models.wideresnet import WideResNet
        model = WideResNet(depth=28, num_classes=10, widen_factor=10)
        model.load_state_dict({k[7:]:v for k,v in torch.load('./checkpoints/rst_adv.pt.ckpt')['state_dict'].items()})

    elif args.net_type == 'overfitting':
        from cifar_models.wideresnet_nosub import WideResNet
        model = WideResNet(depth=34, num_classes=10, widen_factor=10, dropRate=0.0)
        model.load_state_dict({k[7:]:v for k,v in torch.load('./checkpoints/cifar10_wide10_linf_eps8.pth').items()})

    elif args.net_type == 'madry_adv_resnet50':
        from cifar_models.resnet import resnet50
        model = resnet50()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_linf_8.pt')['state_dict'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model)

    elif args.net_type == 'madry_adv_resnet50_l2':
        from cifar_models.resnet import resnet50
        model = resnet50()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_l2_0_5.pt')['state_dict'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model)

    elif args.net_type == 'madry_adv_vgg16':
        from cifar_models.vgg import vgg16
        model = vgg16()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_vgg16_linf_8.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model)

    elif args.net_type == 'adv_pt':
        from cifar_models.wideresnet_pt import WideResNet

        model = WideResNet(28, 10, 10, 0.0)
        model.load_state_dict({k[7:]:v for k,v in torch.load('./checkpoints/cifar10wrn_baseline_epoch_4.pt').items()})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        model = nn.Sequential(normalize, model)

    elif args.net_type == 'jem':
        from cifar_models.jem import CCF, DummyModel, model_attack_wrapper
        f = CCF(28, 10, None)
        ckpt_dict = torch.load('./checkpoints/CIFAR10_MODEL_jem.pt')
        if "model_state_dict" in ckpt_dict:
            # loading from a new checkpoint
            f.load_state_dict(ckpt_dict["model_state_dict"])
        else:
            # loading from an old checkpoint
            f.load_state_dict(ckpt_dict)

        f = DummyModel(f)
        model = model_attack_wrapper(f)

    elif args.net_type == 'madry_adv_inception':
        from cifar_models.inception import inceptionv3
        model = inceptionv3()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_dense_linf8.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model).to(device)
        model.eval()

    elif args.net_type == 'madry_adv_densenet':
        from cifar_models.densenet import densenet121
        model = densenet121()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_inception_linf8.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        model = nn.Sequential(normalize, model).to(device)
        model.eval()
    else:
        raise Exception('The net_type of {} is not supported by now!'.format(args.net_type))

if args.dataset == 'bird_or_bicycle':
    test_loader = bird_or_bicycle.get_iterator('test', batch_size=args.batch_size)

    if args.net_type == 'ResNet50Pre':
        from bird_or_bicycle_models.models_pretrained import ResNet50Pre
        model = ResNet50Pre()
        model.load_state_dict({k[7:]:v for k,v in torch.load('./checkpoints/model-unrestricted-epoch.pt').items()})

if args.dataset == 'imagenet':
    imagenet_val = ImageNet(root_dir='/root/project/data/images', transform = transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(imagenet_val, batch_size=args.batch_size,
                                                shuffle=False, pin_memory=True, num_workers=8)
    args.num_classes = 1000
    if args.net_type == 'denoise_resnet101':
        from imagenet_models.resnet import resnet101_denoise
        model = resnet101_denoise()
        model.load_state_dict(torch.load('./checkpoints/Adv_Denoise_Resnext101.pytorch'), strict=True)
        normalize = NormalizeByChannelMeanStd(
            mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        model = nn.Sequential(normalize, model)

    elif args.net_type == 'adv_resnet50_l2':
        from imagenet_models.resnet_madry import resnet50
        model = resnet50()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/imagenet_l2_3_0.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = nn.Sequential(normalize, model)

    elif args.net_type == 'adv_resnet50_linf4':
        from imagenet_models.resnet_madry import resnet50
        model = resnet50()
        model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/imagenet_linf_4.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        model = nn.Sequential(normalize, model)

    else:
        raise Exception('The net_type of {} is not supported by now!'.format(args.net_type))
    

## load model
# model = resnet50()
# model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_linf_8.pt')['model'].items() if 'attacker' not in k and 'new' not in k})

model.eval()

model = model.to(device)
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

if args.transfer_test:
    if args.sub_net_type == 'madry_adv_vgg16':
        from cifar_models.vgg import vgg16
        sub_model = vgg16()
        sub_model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_vgg16_linf_8.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        sub_model = nn.Sequential(normalize, sub_model).to(device)
        sub_model.eval()
    elif args.sub_net_type == 'madry_adv_resnet50':
        from cifar_models.resnet import resnet50
        sub_model = resnet50()
        sub_model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_linf_8.pt')['state_dict'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        sub_model = nn.Sequential(normalize, sub_model).to(device)
        sub_model.eval()

    elif args.sub_net_type == 'madry_adv_inception':
        from cifar_models.inception import inceptionv3
        sub_model = inceptionv3()
        sub_model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_dense_linf8.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        sub_model = nn.Sequential(normalize, sub_model).to(device)
        sub_model.eval()

    elif args.sub_net_type == 'madry_adv_densenet':
        from cifar_models.densenet import densenet121
        sub_model = densenet121()
        sub_model.load_state_dict({k[13:]:v for k,v in torch.load('./checkpoints/cifar_inception_linf8.pt')['model'].items() if 'attacker' not in k and 'new' not in k})
        normalize = NormalizeByChannelMeanStd(
            mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        sub_model = nn.Sequential(normalize, sub_model).to(device)
        sub_model.eval()

criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)

acc_total = np.ones(len(test_loader.dataset))
acc_curve = []

target_label_list = []
if args.target:
    for loaded_data in test_loader:
        _, test_labels = loaded_data[0], loaded_data[1]

        for i in range(test_labels.size(0)):
            label_choice = list(range(args.num_classes))
            label_choice.remove(test_labels[i].item())
            target_label_list.append(random.choice(label_choice))
    target_label_list = torch.tensor(target_label_list)

# adversary = AutoAttack(model, norm='Linf', eps=8/255, version='standard')
# adversary.attacks_to_run = ['apgd-ce']

for _ in range(args.num_restarts):
    total_num = 0
    clean_acc_num = 0
    adv_acc_num = 0
    attack_successful_num = 0
    
    batch_idx = 0
    for loaded_data in tqdm(test_loader):
        if args.dataset == 'bird_or_bicycle':
            test_images, test_labels, _ = loaded_data[0], loaded_data[1], loaded_data[2]
            test_images = test_images.permute(0,3,1,2)
        else:
            test_images, test_labels = loaded_data[0], loaded_data[1]
            # print(test_images.shape)
            # print(test_labels)
        bstart = batch_idx * args.batch_size
        if test_labels.size(0) < args.batch_size:
            bend = batch_idx * args.batch_size + test_labels.size(0)
        else:
            bend = (batch_idx+1) * args.batch_size
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        total_num += test_labels.size(0)

        clean_logits = model(test_images)
        pred = predict_from_logits(clean_logits)
        pred_right = (pred==test_labels).nonzero().squeeze() 
        if len(target_label_list) != 0:
            target_label = target_label_list[batch_idx*args.batch_size:batch_idx*args.batch_size+test_labels.size(0)].cuda()
            target_label = target_label[pred_right]
        else:
            target_label = None 

        acc_total[bstart:bend] = acc_total[bstart:bend] * (pred==test_labels).cpu().numpy()

        test_images = test_images[pred_right]
        test_labels = test_labels[pred_right]

        if len(test_images.shape) == 3:
            test_images = test_images.unsqueeze(0)
            test_labels = test_labels.unsqueeze(0)
        if len(test_labels.size()) == 0:
            clean_acc_num += 1
        else:
            clean_acc_num += test_labels.size(0)

        # test_images = adversary.run_standard_evaluation(test_images, test_labels, bs=args.batch_size)
        # adv = test_images.clone()

        subpolicy_out_dict = {}

        previous_p = None
        for idx, attacker in enumerate(subpolicy):
            attack_name = attacker['attacker']
            attack_eps = attacker['magnitude']
            attack_steps = attacker['step']
            if idx == 0:
                adv_images, p = apply_attacker(test_images, attack_name, test_labels, model, attack_eps, previous_p, int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=0, target=target_label)
                subpolicy_out_dict[idx] = adv_images.detach()
                pred = predict_from_logits(model(adv_images.detach()))
                if args.target:
                    acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred!=target_label).cpu().numpy()
                else:
                    acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy()

            else:
                ori_adv_images, _ = apply_attacker(test_images, attack_name, test_labels, model, attack_eps, None, int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=0, target=target_label)
                adv_adv_images, p = apply_attacker(subpolicy_out_dict[idx-1], attack_name, test_labels, model, attack_eps, previous_p, int(attack_steps), args.max_epsilon, _type=args.norm, gpu_idx=0, target=target_label)
                
                pred = predict_from_logits(model(ori_adv_images.detach()))
                if args.target:
                    acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred!=target_label).cpu().numpy()
                else:
                    acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy()
                pred = predict_from_logits(model(adv_adv_images.detach()))
                if args.target:
                    acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred!=target_label).cpu().numpy()
                else:
                    acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy()

                subpolicy_out_dict[idx] = adv_adv_images.detach()

                

            if p is not None:
                previous_p = p.detach()
                # print(previous_p.abs().max())
            else:
                previous_p = p
            # print(p.abs().max())
            # print(torch.norm(p.view(p.shape[0], -1), dim=1))

            # pred = predict_from_logits(model(adv_images.detach()))
            # ind_suc = (pred!=test_labels).nonzero().squeeze()
            # adv[ind_suc] = test_images[ind_suc]
            # save_images = test_images[ind_suc]
        # save_images = subpolicy_out_dict[2]
        # if len(save_images.shape) == 3:
        #     save_images = save_images.unsqueeze(0)
        # for i in range(save_images.shape[0]):
        #     save_path = '_'.join([attacker['attacker'] for attacker in subpolicy])
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path, exist_ok=True)
        #     image = np.transpose(save_images[i,:,:,:].cpu().numpy(),(1,2,0))*255
        #     image = image.astype('uint8')
        #     imageio.imwrite(os.path.join(save_path, str(total_num-i)+'.png'), image)

            # if args.transfer_test:
            #     if args.ensemble:
            #         pred = predict_from_logits(sub_model(adv_images.detach()))
            #         acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy()
            # else:
            #     if args.ensemble:
            #         pred = predict_from_logits(model(adv_images.detach()))
            #         acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy()

        # if args.transfer_test:
        #     if not args.ensemble:
        #         pred = predict_from_logits(sub_model(test_images))
        #         acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy()
        # else:
        #     if not args.ensemble:
        #         pred = predict_from_logits(model(test_images))
        #         acc_total[bstart:bend][pred_right.cpu().numpy()] = acc_total[bstart:bend][pred_right.cpu().numpy()] * (pred==test_labels).cpu().numpy()

        # for i in range(test_images.shape[0]):
        #     if pred[i] != test_labels[i,]:
        #         attack_successful_num += 1
        #     else:
        #         adv_acc_num += 1
        batch_idx += 1

        print('accuracy_total: {}/{}'.format(int(acc_total.sum()), len(test_loader.dataset)))
        print('natural_acc_oneshot: ', clean_acc_num/total_num)
        print('robust_acc_oneshot: ', (total_num-len(test_loader.dataset)+acc_total.sum())/total_num)
    acc_curve.append(acc_total.sum())
    print('accuracy_curve: ', acc_curve)