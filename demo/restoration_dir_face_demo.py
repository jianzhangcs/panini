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
        '--imshow', action='store_true', help='whether show image with opencv')
    parser.add_argument('--device', type=int, default=0, help='CUDA device id')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    model = init_model(
        args.config, args.checkpoint, device=torch.device('cuda', args.device))

    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Total number of parameters: %.1f million. '% (num_params / 1000000))
    
    imgs = sorted(os.listdir(args.img_path))
    for i in range(len(imgs)):
        print(os.path.join(args.img_path, imgs[i]))
        output = restoration_inference(model, os.path.join(args.img_path, imgs[i]))
        output = tensor2img(output)

        save_path_i = f'{args.save_path}/{i:05d}.png'

        mmcv.imwrite(output, save_path_i)
        if args.imshow:
            mmcv.imshow(output, 'predicted restoration result')


if __name__ == '__main__':
    main()
