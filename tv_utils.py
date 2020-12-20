from PIL import Image
import PIL
from PIL import ImageFilter
import numbers
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
import torchvision.datasets

_pil_interpolation_to_str = {
    Image.NEAREST: 'PIL.Image.NEAREST',
    Image.BILINEAR: 'PIL.Image.BILINEAR',
    Image.BICUBIC: 'PIL.Image.BICUBIC',
    Image.LANCZOS: 'PIL.Image.LANCZOS',
}

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')


class Permute(nn.Module):

    def __init__(self, permutation = [2,1,0]):
        super().__init__()
        self.permutation = permutation

    def forward(self, input):
        
        return input[:, self.permutation]

class ImageNet(Dataset):
    def __init__(self, root_dir, csv_name='labels', transform=None):
        self.transform = transform
        self.datas = []
        with open(os.path.join(root_dir, csv_name)) as f:
            for line in f.readlines():
                img_path, gt_label = line.strip().split(' ')
                self.datas.append((os.path.join(root_dir, img_path), int(gt_label)))


    def __len__(self):
        l = len(self.datas)
        return l

    def __getitem__(self, idx):
        filename, label_source = self.datas[idx]
        # filename = os.path.join(self.image_dir, self.labels.at[idx, 'ImageId'])
        in_img_t = Image.open(filename)
        if self.transform is not None:
            in_img_t = self.transform(in_img_t)

        return in_img_t, label_source

class GaussianSmoothing(object):
    def __init__(self, radius):
        if isinstance(radius, numbers.Number):
            self.min_radius = radius
            self.max_radius = radius
        elif isinstance(radius, list):
            if len(radius) != 2:
                raise Exception(
                    "`radius` should be a number or a list of two numbers")
            if radius[1] < radius[0]:
                raise Exception(
                    "radius[0] should be <= radius[1]")
            self.min_radius = radius[0]
            self.max_radius = radius[1]
        else:
            raise Exception(
                "`radius` should be a number or a list of two numbers")

    def __call__(self, image):
        radius = np.random.uniform(self.min_radius, self.max_radius)
        return image.filter(ImageFilter.GaussianBlur(radius))

class SpatialAffine(object):

    def __init__(self, degrees, translate=None, scale=None, shear=None, resample=False, fillcolor=0):
        self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
        self.translate = translate
        self.scale = scale
        self.shear = shear
        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = degrees
        if translate is not None:
            max_dx = translate[0]
            max_dy = translate[1]
            translations = (max_dx, max_dy)
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = scale_ranges
        else:
            scale = 1.0

        if shears is not None:
            shear = shears
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, img):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
        return TF.affine(img, *ret, resample=self.resample, fillcolor=self.fillcolor)

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        d['resample'] = _pil_interpolation_to_str[d['resample']]
        return s.format(name=self.__class__.__name__, **d)