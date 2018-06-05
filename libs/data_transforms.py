from __future__ import division
import torch
import random
from PIL import Image, ImageOps, ImageEnhance
import collections
import cv2
import numpy as np
import numbers
from torchvision import datasets, transforms
import torchvision.utils as vutils
import scipy.misc


"""
Add 19 probability masks

transform = EnhancedCompose([
    Merge(),              # merge input and target along the channel axis
    ElasticTransform(),
    RandomRotate(),
    Split([0,1],[1,2]),  # split into 2 images
    [CenterCropNumpy(size=input_shape), CenterCropNumpy(size=target_shape)],
    [NormalizeNumpy(), None],
    [Lambda(to_tensor), Lambda(to_tensor)]
])
"""

class ToTensor(object):
    """Convert an ``numpy.ndarray`` to tensor.
    Converts an numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, img, mask, label):
        """
        Args:
            numpy (numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        img = torch.from_numpy(img.transpose(2,0,1).copy()).float()
        mask = torch.from_numpy(mask.transpose(2,0,1).copy())
        label = torch.from_numpy(label.copy())
        label = label.long()

        return img, mask, label

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Resize(object):
    """Resize the input np array to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (h, w), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size):
        assert isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)
        self.size = size

    def __call__(self, img, mask, label):
        """
        Args:
            img (np array): Image to be scaled.
        Returns:
            np array Image: Rescaled array.
        """
        size = self.size
        if isinstance(size,int):
            # resize short edge to size, keep aspect ratio
            w,h,c = img.shape
            if(w <= h and w == size) or (h <= w and h == size):
                return img, mask, label
            if w < h:
                ow = size
                oh = int(size * h / w)
            else:
                oh = size
                ow = int(size * w / h)
            img = cv2.resize(img, (oh,ow),interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (oh,ow),interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, (oh,ow),interpolation=cv2.INTER_NEAREST)
            return [img, mask, label]
        else:
            img = cv2.resize(img, (self.size[0], self.size[1]), interpolation = cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.size[0], self.size[1]), interpolation = cv2.INTER_NEAREST)
            label = cv2.resize(label, (self.size[0], self.size[1]), interpolation = cv2.INTER_NEAREST)
            return [img, mask, label]

    def __repr__(self):
        # interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + '(size={0}, interpolation={1})'.format(self.size, "BILINEAR")

class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL Image randomly with a given probability.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask, label):
        """
        Args:
            img (np array Image): Image to be flipped.
        Returns:
            np array Image: Randomly flipped image.
        """
        if random.random() < self.p:
            # return F.hflip(img)
            img = img[:,::-1]
            mask = mask[:,::-1]
            label = label[:,::-1]
        return img, mask, label

    def __repr__(self):
        return self.__class__.__name__ + '(p={})'.format(self.p)

class RandomRotation(object):
    """Rotate the image by angle.
    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees).
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        expand (bool, optional): Optional expansion flag.
            If true, expands the output to make it large enough to hold the entire rotated image.
            If false or omitted, make the output image the same size as the input image.
            Note that the expand flag assumes rotation around the center and no translation.
        center (2-tuple, optional): Optional center of rotation.
            Origin is the upper left corner.
            Default is the center of the image.
    """

    def __init__(self, degrees, center=None):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

        self.center = center

    @staticmethod
    def get_params(degrees):
        """Get parameters for ``rotate`` for a random rotation.
        Returns:
            sequence: params to be passed to ``rotate`` for random rotation.
        """
        angle = random.uniform(degrees[0], degrees[1])

        return angle

    def __call__(self, img, mask, label):
        """
            img (cv2 Image): Image to be rotated.
        Returns:
            cv2 Image: Rotated image.
        """
        width, height, _ = img.shape
        angle = self.get_params(self.degrees)
        if self.center is not None:
            M = cv2.getRotationMatrix2D((width/2,height/2), angle, 1)
        else:
            M = cv2.getRotationMatrix2D(self.center, angle, 1)
        img = cv2.warpAffine(img,M,(height,width))
        mask = cv2.warpAffine(mask,M,(height,width),flags=cv2.INTER_NEAREST)
        label = cv2.warpAffine(label,M,(height,width),flags=cv2.INTER_NEAREST,borderValue=255)
        return img, mask, label

    def __repr__(self):
        format_string = self.__class__.__name__ + '(degrees={0}'.format(self.degrees)
        if self.center is not None:
            format_string += ', center={0}'.format(self.center)
        format_string += ')'
        return format_string

class RandomScale(object):
    def __init__(self, scale):
        if isinstance(scale, numbers.Number):
            scale = [1 / scale, scale]
        self.scale = scale

    def __call__(self, image, mask, label):
        ratio = random.uniform(self.scale[0], self.scale[1])

        h, w, c = image.shape
        tw = int(ratio * w)
        th = int(ratio * h)
        if ratio == 1:
            return image, mask, label
        image_resize = cv2.resize(image,(tw,th), interpolation=cv2.INTER_AREA)
        mask_resize = cv2.resize(mask,(tw,th), interpolation=cv2.INTER_NEAREST)
        label_resize = cv2.resize(label,(tw,th), interpolation=cv2.INTER_NEAREST)
        
        return image_resize, mask_resize, label_resize

class ConstrainedCrop(object):
    """
    crop input patches according to label distribution.

    args:

    """
    def __init__(self, size):
        super(ConstrainedCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        #self.valid_cls = np.array([1,3,4,5,6,7,9,11,12,13,14,15,16,17,18])

    def get_params(self, lab, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            lab (PIL image): image/label to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h,w = lab.shape
        th, tw = output_size
        if w == tw and h==th:
            return 0,0,h,w

        count = 0
        protect = 0
        if len(np.unique(lab[lab<22])) < 2:
            mmp = 0
        else:
            mmp = 1
        while True:
            i = random.randint(0, h-th)
            j = random.randint(0, w-tw)
            tmp_lb = lab[i:i+th, j:j+th]
            nidx = np.unique(tmp_lb[tmp_lb<21])
            if (len(nidx) < 2 and mmp == 1) and protect < 10:
                protect += 1
                continue
            else:
                break
        return i, j, th, tw

    def __call__(self, img, mask, label):
        """
        Args:
            img (cv2 Image): Image to be cropped.
            mask (numpy #num_classes channels labels): Mask to be cropped
            lab (cv2 signle channel labels): Label to be cropped
        Returns:
            cv2 Image: Cropped image.
            numpy array Mask: Cropped mask.
            cv2 Label: Cropped label.
        """
        i, j, h, w = self.get_params(label, self.size)
        img_crop = img[i:i+h,j:j+w]
        mask_crop = mask[i:i+h,j:j+w]
        lab_crop = label[i:i+h,j:j+w]

        return img_crop, mask_crop, lab_crop

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}'.format(self.size)

class ConstrainedCropCityScapes(object):
    """
    crop input patches according to label distribution.

    args:

    """
    def __init__(self, size):
        super(ConstrainedCropCityScapes, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.valid_cls = np.array([1,3,4,5,6,7,9,11,12,13,14,15,16,17,18])

    def get_params(self, lab, output_size):
        """Get parameters for ``crop`` for a random crop.
        Args:
            lab (PIL image): image/label to be cropped.
            output_size (tuple): Expected output size of the crop.
        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for random crop.
        """
        h,w = lab.shape
        th, tw = output_size
        if w == tw and h==th:
            return 0,0,h,w

        count = 0
        project = 0
        if len(np.unique(lab[lab<19])) < 2 or len(np.intersect1d(np.unique(lab[lab<19]), self.valid_cls)) < 1:
            mmp = 0
        else:
            mmp = 1
        while True:
            i = random.randint(0, h-th)
            j = random.randint(0, w-tw)
            tmp_lb = lab[i:i+th, j:j+th]
            nidx = np.unique(tmp_lb[tmp_lb<19])
            if ((len(nidx)<2 and len(np.intersect1d(np.unique(tmp_lb[tmp_lb<19]), self.valid_cls)) < 1) and mmp) and project < 5:
                project += 1
                continue
            else:
                break
        return i, j, th, tw

    def __call__(self, img, mask, label):
        """
        Args:
            img (cv2 Image): Image to be cropped.
            mask (numpy #num_classes channels labels): Mask to be cropped
            lab (cv2 signle channel labels): Label to be cropped
        Returns:
            cv2 Image: Cropped image.
            numpy array Mask: Cropped mask.
            cv2 Label: Cropped label.
        """
        i, j, h, w = self.get_params(label, self.size)
        img_crop = img[i:i+h,j:j+w]
        mask_crop = mask[i:i+h,j:j+w]
        lab_crop = label[i:i+h,j:j+w]

        return img_crop, mask_crop, lab_crop

    def __repr__(self):
        return self.__class__.__name__ + '(size={0}'.format(self.size)

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self,img,mask,label):
        for t in self.transforms:
            img, mask, label = t(img, mask, label)
        return img, mask, label

class Normalize(object):
    """Given mean: (B, G, R) and std: (B, G, R),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    """

    def __init__(self, mean):
        self.mean = torch.FloatTensor(mean)
        #self.std = torch.FloatTensor(std)

    def __call__(self, image, mask, label):
        # RGB -> BRG
        image = image[[2,1,0], :, :]
        for t, m in zip(image, self.mean):
            t.sub_(m)
        return image, mask, label

class pad(object):
    """pad the input with zeros"""
    def __init__(self, loadsize):
        super( pad, self).__init__()
        self.loadsize = loadsize

    def __call__(self, image, mask, label):
        h,w = label.shape
        pad_h = int((self.loadsize-h)/2)
        pad_w = int((self.loadsize-w)/2)
        pad_h_2 = self.loadsize-h-pad_h
        pad_w_2 = self.loadsize-w-pad_w
        image = cv2.copyMakeBorder(image, pad_h, pad_h_2, pad_w, pad_w_2, cv2.BORDER_CONSTANT, 0)
        mask = cv2.copyMakeBorder(mask, pad_h, pad_h_2, pad_w, pad_w_2, cv2.BORDER_CONSTANT, 0)
        # label: do nothing!
        return image, mask, label