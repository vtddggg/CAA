# CAA
This is the Pytorch implementation of our AAAI2021 paper [Composite Adversarial Attacks](https://arxiv.org/abs/2012.05434)

## Pre-requisites
- torch >= 1.3.0

- torchvision

- advertorch

- tqdm

- pillow

- imagenet_c (Only for unrestricted adversarial attacks)


## Usage

### Linf Attack
Collect [adv. training models](https://www.dropbox.com/s/c9qlt1lbdnu9tlo/cifar_linf_8.pt?dl=0) and place it to `checkpoints`. Then run 
```
python test_attacker.py --batch_size 512 --dataset cifar10 --net_type madry_adv_resnet50 --norm linf
```

### Unrestricted Attack
The code is only for test on bird_or_bicycle datasets, also you can adapt it to your own datasets and tasks. For those who want to run on bird_or_bicycle datasets, you must first install `bird_or_bicycle`:
```
git clone https://github.com/google/unrestricted-adversarial-examples
pip install -e unrestricted-adversarial-examples/bird-or-bicycle

bird-or-bicycle-download
```
Then collect unrestricted defense models in [Unrestricted Adversarial Examples Challenge](https://github.com/openphilanthropy/unrestricted-adversarial-examples), such as [TRADESv2](https://github.com/xincoder/google_attack), and place it to `checkpoints`. Finally, run 
```
python test_attacker.py --batch_size 12 --dataset bird_or_bicycle --net_type ResNet50Pre --norm unrestricted
```

### L2 Attack
Collect [adv. training models](https://www.dropbox.com/s/1zazwjfzee7c8i4/cifar_l2_0_5.pt?dl=0) and place it to `checkpoints`. Then run 
```
python test_attacker.py --batch_size 384 --dataset cifar10 --net_type madry_adv_resnet50_l2 --norm l2 --max_epsilon 0.5
```
*For this implementation, the codes are not passing a fully test for L2 Attack due to the much long elapsed time, so there may exists some bugs in L2 attack case*

### Define a custom attack
Your can define arbitrary attack policies by giving a list of attacker dict:

For example, if you want to compose `MultiTargetedAttack`, `MultiTargetedAttack` and `CWLinf_Attack_adaptive_stepsize`, just define the list like follows:
```
[{'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 50}, {'attacker': 'MultiTargetedAttack', 'magnitude': 8/255, 'step': 25}, {'attacker': 'CWLinf_Attack_adaptive_stepsize', 'magnitude': 8/255, 'step': 125}]
```
CAA also supports single attacker without any combination: when you give a list like `[{'attacker': 'ODI_Step_stepsize', 'magnitude': 8/255, 'step': 150}]`, it means you are runing a single `ODI-PGD` attack.

Now this code supports attackers as follows:
- GradientSignAttack
- PGD_Attack_adaptive_stepsize
- MI_Attack_adaptive_stepsize
- CWLinf_Attack_adaptive_stepsize
- MultiTargetedAttack
- ODI_Cos_stepsize
- ODI_Cyclical_stepsize
- ODI_Step_stepsize
- CWL2Attack
- DDNL2Attack
- GaussianBlurAttack
- GaussianNoiseAttack
- ContrastAttack
- SaturateAttack
- ElasticTransformAttack
- JpegCompressionAttack
- ShotNoiseAttack
- ImpulseNoiseAttack
- DefocusBlurAttack
- GlassBlurAttack
- MotionBlurAttack
- ZoomBlurAttack
- FogAttack
- BrightnessAttack
- PixelateAttack
- SpeckleNoiseAttack
- SpatterAttack
- SPSAAttack
- SpatialAttack

## Warning

The codes have not been carefully arranged and still been messy now. If you meet any bugs, feel free to report in issues. 

## Citations
```
@article{brown2018unrestricted,
  title={Unrestricted adversarial examples},
  author={Brown, Tom B and Carlini, Nicholas and Zhang, Chiyuan and Olsson, Catherine and Christiano, Paul and Goodfellow, Ian},
  journal={arXiv preprint arXiv:1809.08352},
  year={2018}
}
```

```
@article{ding2019advertorch,
  title={AdverTorch v0. 1: An adversarial robustness toolbox based on pytorch},
  author={Ding, Gavin Weiguang and Wang, Luyu and Jin, Xiaomeng},
  journal={arXiv preprint arXiv:1902.07623},
  year={2019}
}
```

```
@article{rauber2017foolbox,
  title={Foolbox: A python toolbox to benchmark the robustness of machine learning models},
  author={Rauber, Jonas and Brendel, Wieland and Bethge, Matthias},
  journal={arXiv preprint arXiv:1707.04131},
  year={2017}
}
```

```
@article{papernot2016technical,
  title={Technical report on the cleverhans v2. 1.0 adversarial examples library},
  author={Papernot, Nicolas and Faghri, Fartash and Carlini, Nicholas and Goodfellow, Ian and Feinman, Reuben and Kurakin, Alexey and Xie, Cihang and Sharma, Yash and Brown, Tom and Roy, Aurko and others},
  journal={arXiv preprint arXiv:1610.00768},
  year={2016}
}
```

```
@article{croce2020reliable,
  title={Reliable evaluation of adversarial robustness with an ensemble of diverse parameter-free attacks},
  author={Croce, Francesco and Hein, Matthias},
  journal={arXiv preprint arXiv:2003.01690},
  year={2020}
}
```

```
@article{tashiro2020output,
  title={Output Diversified Initialization for Adversarial Attacks},
  author={Tashiro, Yusuke and Song, Yang and Ermon, Stefano},
  journal={arXiv preprint arXiv:2003.06878},
  year={2020}
}
```
