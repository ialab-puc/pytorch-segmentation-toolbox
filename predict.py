import argparse
import scipy
from scipy import ndimage
import cv2
import numpy as np
import sys
import json

import torch

from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data
import networks
from dataset.datasets import CSDataTestSet
from collections import OrderedDict
import os
import scipy.ndimage as nd
from math import ceil
from PIL import Image as PILImage
from utils.pyt_utils import load_model
from evaluate import get_palette, predict_multiscale, predict_sliding, get_confusion_matrix, predict_whole
from engine import Engine
import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = 'cityscapes'
DATA_LIST_PATH = './dataset/list/cityscapes/val.lst'
IGNORE_LABEL = 255
NUM_CLASSES = 19
NUM_STEPS = 500 # Number of images in the validation set.
INPUT_SIZE = '340,480'
RESTORE_FROM = './deeplab_resnet.ckpt'
BATCH_SIZE = 8

def get_parser():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=str, default='0',
                        help="choose gpu device.")
    parser.add_argument("--recurrence", type=int, default=1,
                        help="choose the number of recurrence.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    return parser


def main():
    """Create the model and start the evaluation process."""
    parser = get_parser()

    with Engine(custom_parser=parser) as engine:
        args = parser.parse_args()

        cudnn.benchmark = True
        h, w = map(int, args.input_size.split(','))
        input_size = (h, w)

        seg_model = networks.pspnet.Seg_Model(
            num_classes=args.num_classes
        )

        load_model(seg_model, args.restore_from)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seg_model.to(device)

        model = engine.data_parallel(seg_model)
        model.eval()

        dataset = data.DataLoader(CSDataTestSet(args.data_dir, args.data_list, crop_size=input_size, mean=IMG_MEAN),
                                        batch_size=1, shuffle=False, pin_memory=True)
        test_loader, test_sampler = engine.get_test_loader(dataset)

        if engine.distributed:
            test_sampler.set_epoch(0)

        palette = get_palette(NUM_CLASSES)
        # interp = nn.Upsample(size=(1024, 2048), mode='bilinear', align_corners=True)

        if not os.path.exists('outputs'):
            os.makedirs('outputs')

        for index, batch in enumerate(dataset):
            if index % 100 == 0:
                print('%d processd'%(index))
            image, name, size = batch
            size = size[0].numpy()
            with torch.no_grad():
                # output = predict_sliding(model, image.numpy(), input_size, args.num_classes, True, args.recurrence)
                output = predict_whole(model, image.numpy(), input_size, args.recurrence)
            seg_pred = np.asarray(np.argmax(output, axis=2), dtype=np.uint8)
            output_im = PILImage.fromarray(seg_pred)
            output_im.putpalette(palette)
            # output_im = output_im.crop((0, 0, w, h))
            output_im.save('outputs/'+name[0]+'.png')

if __name__ == '__main__':
    main()
