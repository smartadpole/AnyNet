from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
from models.anynet import AnyNet
from PIL import Image
from collections import OrderedDict

# 2012 data /media/jiaren/ImageNet/data_scene_flow_2012/testing/

parser = argparse.ArgumentParser(description='PSMNet')
parser.add_argument('--KITTI', default='2015',
                    help='KITTI version')
parser.add_argument('--datapath', default='/media/jiaren/ImageNet/data_scene_flow_2015/testing/',
                    help='select model')
parser.add_argument('--loadmodel', default='./trained/pretrained_model_KITTI2015.tar',
                    help='loading model')
parser.add_argument('--leftimg', default='./image/left.png',
                    help='load model')
parser.add_argument('--rightimg', default='./image/right.png',
                    help='load model')
parser.add_argument('--maxdisp', type=int, default=192,
                    help='maxium disparity')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--max_disparity', type=int, default=192)
parser.add_argument('--maxdisplist', type=int, nargs='+', default=[12, 3, 3])
parser.add_argument('--with_spn', action='store_true', help='with spn network or not')
parser.add_argument('--init_channels', type=int, default=1, help='initial channels for 2d feature extractor')
parser.add_argument('--nblocks', type=int, default=2, help='number of layers in each stage')
parser.add_argument('--channels_3d', type=int, default=4, help='number of initial channels 3d feature extractor ')
parser.add_argument('--layers_3d', type=int, default=4, help='number of initial layers in 3d network')
parser.add_argument('--growth_rate', type=int, nargs='+', default=[4,1,1], help='growth rate in the 3d network')
parser.add_argument('--spn_init_channels', type=int, default=8, help='initial channels for spnet')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

model = AnyNet(args)

model = nn.DataParallel(model, device_ids=[0])
model.cuda()

if args.loadmodel is not None:
    print('load AnyNet')
    checkpoint = torch.load(args.loadmodel)
    model.load_state_dict(checkpoint['state_dict'], strict=False)

print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))

from torch.onnx import register_custom_op_symbolic
import torch.onnx.symbolic_helper as sym_help

def grid_sampler(g, input, grid, mode, padding_mode, align_corners):
    # mode
    #   'bilinear'      : onnx::Constant[value={0}]
    #   'nearest'       : onnx::Constant[value={1}]
    #   'bicubic'       : onnx::Constant[value={2}]
    # padding_mode
    #   'zeros'         : onnx::Constant[value={0}]
    #   'border'        : onnx::Constant[value={1}]
    #   'reflection'    : onnx::Constant[value={2}]
    mode = sym_help._maybe_get_const(mode, "i")
    padding_mode = sym_help._maybe_get_const(padding_mode, "i")
    mode_str = ['bilinear', 'nearest', 'bicubic'][mode]
    padding_mode_str = ['zeros', 'border', 'reflection'][padding_mode]
    align_corners = int(sym_help._maybe_get_const(align_corners, "b"))

    return g.op("com.microsoft::GridSample", input, grid,
                mode_s=mode_str,
                padding_mode_s=padding_mode_str,
                align_corners_i=align_corners)


def test(imgL, imgR):
    model.eval()

    if args.cuda:
        imgL = imgL.cuda()
        imgR = imgR.cuda()

    with torch.no_grad():
        disp = model(imgL, imgR)

    disp = torch.squeeze(disp[0])
    pred_disp = disp.data.cpu().numpy()

    onnx_input_L = torch.rand(1, 3, 400, 640)
    onnx_input_R = torch.rand(1, 3, 400, 640)
    onnx_input_L = onnx_input_L.to("cuda:0")
    onnx_input_R = onnx_input_R.to("cuda:0")
    torch.onnx.export(model.module,
                      (imgL, imgR),
                      "anynet.onnx",
                      # where to save the model (can be a file or file-like object)
                      export_params=True,  # store the trained parameter weights inside the model file
                      opset_version=12,  # the ONNX version to export the model to
                      do_constant_folding=True,  # whether to execute constant folding for optimization
                      input_names=['left', 'right'],  # the model's input names
                      output_names=['output'])

    return pred_disp


def main():
    normal_mean_var = {'mean': [0.485, 0.456, 0.406],
                       'std': [0.229, 0.224, 0.225]}
    infer_transform = transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(**normal_mean_var)])

    imgL_o = Image.open(args.leftimg).convert('RGB')
    imgR_o = Image.open(args.rightimg).convert('RGB')

    imgL = infer_transform(imgL_o)
    imgR = infer_transform(imgR_o)

    # pad to width and hight to 16 times
    if imgL.shape[1] % 16 != 0:
        times = imgL.shape[1] // 16
        top_pad = (times + 1) * 16 - imgL.shape[1]
    else:
        top_pad = 0

    if imgL.shape[2] % 16 != 0:
        times = imgL.shape[2] // 16
        right_pad = (times + 1) * 16 - imgL.shape[2]
    else:
        right_pad = 0

    imgL = F.pad(imgL, (0, right_pad, top_pad, 0)).unsqueeze(0)
    imgR = F.pad(imgR, (0, right_pad, top_pad, 0)).unsqueeze(0)

    start_time = time.time()
    pred_disp = test(imgL, imgR)


if __name__ == '__main__':
    main()
