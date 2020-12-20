'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ResNet18Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self):
        super(ResNet18Pre, self).__init__()
        self.res18 = torchvision.models.resnet18(pretrained=True)
        num_ftrs = self.res18.fc.in_features
        self.res18.fc = nn.Linear(num_ftrs, 2)
        for param in self.res18.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.res18(x)
        return x


class ResNet34Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self):
        super(ResNet34Pre, self).__init__()
        self.res34 = torchvision.models.resnet34(pretrained=True)
        num_ftrs = self.res34.fc.in_features
        self.res34.fc = nn.Linear(num_ftrs, 2)
        for param in self.res34.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)


    def forward(self, x):
        x = self.upsample(x)
        x = self.res34(x)
        return x


class ResNet50Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self):
        super(ResNet50Pre, self).__init__()
        self.res50 = torchvision.models.resnet50(pretrained=True)
        num_ftrs = self.res50.fc.in_features
        self.res50.fc = nn.Linear(num_ftrs, 2)
        for param in self.res50.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.res50(x)
        return x


class ResNet101Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self):
        super(ResNet101Pre, self).__init__()
        self.res101 = torchvision.models.resnet101(pretrained=True)
        num_ftrs = self.res101.fc.in_features
        self.res101.fc = nn.Linear(num_ftrs, 2)
        for param in self.res101.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.res101(x)
        return x


class ResNet152Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self):
        super(ResNet152Pre, self).__init__()
        self.res152 = torchvision.models.resnet152(pretrained=True)
        num_ftrs = self.res152.fc.in_features
        self.res152.fc = nn.Linear(num_ftrs, 2)
        for param in self.res152.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.res152(x)
        return x



class DenseNet121Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self):
        super(DenseNet121Pre, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Linear(num_ftrs, 2)
        for param in self.densenet121.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.densenet121(x)
        return x


class DenseNet161Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self):
        super(DenseNet161Pre, self).__init__()
        self.densenet161 = torchvision.models.densenet161(pretrained=True)
        num_ftrs = self.densenet161.classifier.in_features
        self.densenet161.classifier = nn.Linear(num_ftrs, 2)
        for param in self.densenet161.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.densenet161(x)
        return x


class VGG11Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self):
        super(VGG11Pre, self).__init__()
        self.vgg11_bn = torchvision.models.vgg11_bn(pretrained=True)
        self.vgg11_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        for param in self.vgg11_bn.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.vgg11_bn(x)
        return x


class VGG13Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self):
        super(VGG13Pre, self).__init__()
        self.vgg13_bn = torchvision.models.vgg13_bn(pretrained=True)
        self.vgg13_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        for param in self.vgg13_bn.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.vgg13_bn(x)
        return x


class VGG19Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self):
        super(VGG19Pre, self).__init__()
        self.vgg19_bn = torchvision.models.vgg19_bn(pretrained=True)
        self.vgg19_bn.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 2),
        )
        for param in self.vgg19_bn.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=224, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.vgg19_bn(x)
        return x


class InceptionV3Pre(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """

    def __init__(self):
        super(InceptionV3Pre, self).__init__()
        self.inception_v3 = torchvision.models.inception_v3(pretrained=True)
        self.inception_v3.fc = nn.Linear(2048, 2)
        self.inception_v3.aux_logits = False
        for param in self.inception_v3.parameters():
            param.requires_grad = True
        # unsample image
        self.upsample = nn.Upsample(size=299, mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.inception_v3(x)
        return x