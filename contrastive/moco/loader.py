# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from PIL import ImageFilter
import random
import numpy as np


def cropmix(view1, view2):
    def random_bbox(lam, H, W):
        cut_rat = np.sqrt(1.0 - lam)
        cut_w = np.int(W * cut_rat)
        cut_h = np.int(H * cut_rat)

        cx = np.random.randint(W)
        cy = np.random.randint(H)

        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        return bbx1, bby1, bbx2, bby2

    _, h, w = view1.shape
    lam = np.random.uniform(low=0.0, high=1.0)
    bbx1, bby1, bbx2, bby2 = random_bbox(lam, h, w)
    view1[:, bbx1:bbx2, bby1:bby2] = view2[:, bbx1:bbx2, bby1:bby2]
    return view1


class CropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, key_transform, query_mini_transform, query_transform, enable_cropmix,
                 enable_multicrop, enable_mean_encoding, query_transform_small, query_transform_large):
        self.key_transform = key_transform
        self.query_mini_transform = query_mini_transform
        self.query_transform = query_transform
        self.enable_cropmix = enable_cropmix
        self.enable_multicrop = enable_multicrop
        self.enable_mean_encoding = enable_mean_encoding
        self.query_transform_small = query_transform_small
        self.query_transform_large = query_transform_large

    def __call__(self, x):
        crops = []
        # Query crop
        if self.enable_cropmix:
            q = cropmix(self.query_transform_small(x), self.query_transform_large(x), )
        else:
            q = self.query_transform(x)
        crops.append(q)
        # Query mini crops
        if self.enable_multicrop:
            for i in range(6):
                crops.append(self.query_mini_transform(x))
        # Key crop
        crops.append(self.key_transform(x))
        if self.enable_mean_encoding:
            crops.append(self.key_transform(x))
        return crops


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
