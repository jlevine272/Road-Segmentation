#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import math
import os
from os.path import exists, join, split
import threading

import time

import numpy as np
import shutil

import sys
from PIL import Image
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import drn
import data_transforms as transforms
from segment import DRNSeg

try:
    from modules import batchnormsync
except ImportError:
    pass


CITYSCAPE_PALETTE = np.asarray([
    [128, 64, 128],
    [244, 35, 232],
    [70, 70, 70],
    [102, 102, 156],
    [190, 153, 153],
    [153, 153, 153],
    [250, 170, 30],
    [220, 220, 0],
    [107, 142, 35],
    [152, 251, 152],
    [70, 130, 180],
    [220, 20, 60],
    [255, 0, 0],
    [0, 0, 142],
    [0, 0, 70],
    [0, 60, 100],
    [0, 80, 100],
    [0, 0, 230],
    [119, 11, 32],
    [0, 0, 0]], dtype=np.uint8)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def generate_colorful_images(predictions):
    return [ Image.fromarray(CITYSCAPE_PALETTE[predictions[i].squeeze()]) for i in range(len(predictions)) ]

def generate_road_mask(predictions):
    return [Image.fromarray(predictions[i].squeeze() == 0) for i in range(len(predictions))]


def semantic_segmentation(eval_data_loader, model, output_ims=True):
    """
    Produce Segmented Outputs for images
    @param eval_data_loader: a SegList that outputs test images. Should have Batch size of 1
    @param model: a trained DRNSeg model
    @param output_ims: Determines whether the function will return the segmented images
    @return: tuple of (mAP, outputs, road_mask), where mAP is a float and outputs is a list of segmented images.
                if has_gt is false, mAP will be replaced with None. If output_ims=False,
                outputs will be an empty list. road_mask
    """
    model.eval()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    end = time.time()
    output = []
    masks = []
    inputs = []
    for iter, (image, rgb_image) in enumerate(eval_data_loader):
        data_time.update(time.time() - end)
        image_var = Variable(image, requires_grad=False, volatile=True)

        # Get Model results
        final = model(image_var)[0]
        # Classes
        _, pred = torch.max(final, 1)
        pred = pred.cpu().data.numpy()
        batch_time.update(time.time() - end)
        if output_ims:
            output.extend(generate_colorful_images(pred))
            masks.extend(generate_road_mask(pred))
            inputs.append(rgb_image[0])
        end = time.time()
    # The first class is the road
    return None, output, masks, inputs

