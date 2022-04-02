# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch

import sys
sys.path.append(".")
sys.path.append("..")

from mmedit.apis import init_model, restoration_inference
from mmedit.core import tensor2img

def parse_args():
    parser = argparse.ArgumentParser(description='Restoration demo')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument('--img_path', help='path to input image file')
    parser.add_argument('--save_path', help='path to save restoration result')
    parser.add_argument(
        '--upscale_factor',
        type=int,
        default=1,
        help='the number of times the input image is upsampled.')
    parser.add_argument(
        '--face_size',
        type=int,
        default=512,
        help='the size of the cropped and aligned faces..')
    parser.add_argument(
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if not os.path.isfile(args.img_path):
        raise ValueError('It seems that you did not input a valid '
                         '"image_path". Please double check your input, or '
                         'you may want to use "restoration_video_demo.py" '
                         'for video restoration.')

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters: %.1f million. '% (num_params / 1000000))
    
    print(args.img_path)
    output = restoration_inference(model, args.img_path)
    output = tensor2img(output)

    mmcv.imwrite(output, args.save_path)
    if args.imshow:
        mmcv.imshow(output, 'predicted restoration result')


if __name__ == '__main__':
    main()