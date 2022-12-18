from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F


class MyUpsampling2d(nn.Module):
    def __init__(self, size=None, scale_factor=None):
        super(MyUpsampling2d, self).__init__()
        self.size = size
        self.scale_factor = scale_factor
    
    def forward(self, x):
        if self.size is not None:
            return F.interpolate(x, size=self.size, mode='bilinear', align_corners=False)
        elif self.scale_factor is not None:
            return F.interpolate(x, scale_factor=self.scale_factor, mode='bilinear', align_corners=False)
        else:
            raise ValueError('size or scale_factor must be defined')


def preconv2d(in_planes, out_planes, kernel_size, stride, pad, dilation=1, bn=True):
    if bn:
        return nn.Sequential(
            nn.BatchNorm2d(in_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation=dilation, bias=False))

class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        if is_deconv:
            self.up = nn.Sequential(
                nn.BatchNorm2d(in_size),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, padding=0)
            )
        else:
            self.up = MyUpsampling2d(scale_factor=2)
            in_size = int(in_size * 1.5)

        self.conv = nn.Sequential(
            preconv2d(in_size, out_size, 3, 1, 1),
            preconv2d(out_size, out_size, 3, 1, 1),
        )

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        buttom, right = inputs1.size(2)%2, inputs1.size(3)%2
        outputs2 = F.pad(outputs2, (0, -right, 0, -buttom))
        return self.conv(torch.cat([inputs1, outputs2], 1))


class GateRecurrent(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, X: torch.Tensor, G1, G2, G3):
        width = X.size(3)
        H = torch.zeros_like(X)
        for t in range(width):
            g1 = G1[..., t]
            g2 = G2[..., t]
            g3 = G3[..., t]

            h1 = F.pad(H[:, :, :-1, t-1], (1, 0))
            h2 = H[:, :, :, t-1]
            h3 = F.pad(H[:, :, 1:, t-1], (0, 1))

            g = g1 + g2 + g3
            x = X[..., t]
            H[..., t] = (1 - g) * x + (h1 * g1 + h2 * g2 + h3 * g3)

        return H


class feature_extraction_conv(nn.Module):
    def __init__(self, init_channels,  nblock=2):
        super(feature_extraction_conv, self).__init__()

        self.init_channels = init_channels
        nC = self.init_channels
        downsample_conv = [nn.Conv2d(3,  nC, 3, 1, 1), # 512x256
                                    preconv2d(nC, nC, 3, 2, 1)]
        downsample_conv = nn.Sequential(*downsample_conv)

        inC = nC
        outC = 2*nC
        block0 = self._make_block(inC, outC, nblock)
        self.block0 = nn.Sequential(downsample_conv, block0)

        nC = 2*nC
        self.blocks = []
        for i in range(2):
            self.blocks.append(self._make_block((2**i)*nC,  (2**(i+1))*nC, nblock))

        self.upblocks = []
        for i in reversed(range(2)):
            self.upblocks.append(unetUp(nC*2**(i+1), nC*2**i, False))

        self.blocks = nn.ModuleList(self.blocks)
        self.upblocks = nn.ModuleList(self.upblocks)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_block(self, inC, outC, nblock ):
        model = []
        model.append(nn.MaxPool2d(2,2))
        for i in range(nblock):
            model.append(preconv2d(inC, outC, 3, 1, 1))
            inC = outC
        return nn.Sequential(*model)


    def forward(self, x):
        downs = [self.block0(x)]
        for i in range(2):
            downs.append(self.blocks[i](downs[-1]))
        downs = list(reversed(downs))
        for i in range(1,3):
            downs[i] = self.upblocks[i-1](downs[i], downs[i-1])
        return downs



def batch_relu_conv3d(in_planes, out_planes, kernel_size=3, stride=1, pad=1, bn3d=True):
    if bn3d:
        return nn.Sequential(
            nn.BatchNorm3d(in_planes),
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))
    else:
        return nn.Sequential(
            nn.ReLU(),
            nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride, bias=False))

def post_3dconvs(layers, channels):
    net  = [batch_relu_conv3d(1, channels)]
    net += [batch_relu_conv3d(channels, channels) for _ in range(layers)]
    net += [batch_relu_conv3d(channels, 1)]
    return nn.Sequential(*net)

