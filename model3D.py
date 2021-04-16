
import time

import numpy as np
import cv2
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import torch
import torchvision
from torch import Tensor, nn
from torch.nn import functional as F
import kornia
import Dataset3D

test = 0
def build_conv_layer(fi, fo, bn=True, k=3, upscale=False, pool=False):
    #  standard conv layer
    return nn.Sequential(
        nn.UpsamplingBilinear2d(scale_factor=2) if upscale else nn.Identity(),
        nn.Conv2d(fi, fo, kernel_size=k, padding=k//2),
        nn.BatchNorm2d(fo) if bn else nn.Identity(),
        nn.MaxPool2d(2, 2) if pool else nn.Identity(),
    )

class Model(nn.Module):
    def __init__(self, res=128, left=(3, 16, 32, 64), right=(64, 64, 32), n_objs=3, smooth_cs=True):
        super().__init__()
        right = *right, n_objs * 8
        assert len(left) == len(right)
        self.res = res
        self.n_objs = n_objs
        self.smooth_cs = smooth_cs
        layers = [build_conv_layer(fi, fo, pool=True) for fi, fo in zip(left, left[1:])]
        layers += [build_conv_layer(fi, fo, upscale=True) for fi, fo in zip(right, right[1:-1])]
        layers.append(build_conv_layer(right[-2], right[-1], upscale=True, bn=False))
        self.layers = nn.Sequential(*layers)

        coords = np.linspace(-.5, .5, res, dtype=np.float32)
        coords = np.stack(np.meshgrid(coords, coords))  # (x, y) coordinates!
        self.coords = nn.Parameter(torch.from_numpy(coords), False)

        self.eval()
        assert self.layers(torch.zeros(1, left[0], res, res)).shape == (1, n_objs * 8, res, res)
        self.train()

    def forward(self,x_a ):#, x_b):
        r = self.res
        x_a = x_a.float()
        assert x_a.shape[1:] == (3, r, r), x_a.shape
        B = x_a.shape[0]

        y_cam_a = self.layers(x_a)

        heatmaps_a, cs_raw_a = y_cam_a[:, :self.n_objs*2], y_cam_a[:, self.n_objs*2:]

        cs_raw_a = cs_raw_a.view(B, self.n_objs, 6, r, r)
        print(heatmaps_a.shape)
        heatmaps_a = F.softmax(heatmaps_a.view(B, self.n_objs*2, -1), dim=-1).view(B, self.n_objs, 2, r, r)
        print(heatmaps_a.shape)
        coords_a = heatmaps_a * self.coords.view(1, 1, 2, r, r)  # B, n_objs, 4, r, r
        print(coords_a.shape)
        coords_a = coords_a.sum(dim=(-1, -2))  # B, n_objs*2, 2

        if self.smooth_cs:
            cs_a = torch.sum(heatmaps_a.detach() * cs_raw_a, dim=(-1, -2)) # B, n_objs, 2
        else:
            # there must be a better way than this crazy indexing?

            # idxs = coords as index in heatmaps (0 to 126)
            idxs = torch.clamp_(torch.round_((coords_a + 0.5) * r).long(), 0, r-1)  # B, n_objs, 2  (xy)
            print(coords_a.shape)
            print(idxs.shape)
            print(cs_raw_a.shape)
            print()
            cs_a = cs_raw_a[
                torch.arange(B).view((B, 1, 1)),                    # Take from all batches
                torch.arange(self.n_objs).view(1, self.n_objs, 1),  # for all objects
                torch.arange(6).view(1, 1, 6),                      #  sin and cos
                idxs[..., 1].view(B, self.n_objs, 1),               # at ys
                idxs[..., 0].view(B, self.n_objs, 1),               # and xs from idxs
            ]



        return coords_a, cs_a, heatmaps_a, cs_raw_a


def train(setup_path="/home/markus/Documents/GitHub/Unsupervised-pose-estimation-from-RGB-images/setups.txt", img_path ="/home/markus/Documents/GitHub/Unsupervised-pose-estimation-from-RGB-images/3DdatasetImgs/" ):
    res = 128
    n_epochs = 200
    epoch_size = 1000
    device = torch.device('cuda')

    dataset = Dataset3D.Dataset(1000, img_path=img_path, setup_path=setup_path)

    worker_init_fn = lambda *_: np.random.seed()
    dataloader = torch.utils.data.DataLoader(dataset, 32, num_workers=4, worker_init_fn=worker_init_fn)
    model_1 = Model(res=res, smooth_cs=False).to(device)
    model_2 = Model(res=res, smooth_cs=False).to(device)
    opt = torch.optim.Adam(model_1.parameters(), lr=1e-3)

    pbar = tqdm(total=n_epochs * len(dataloader))
    losses = []

    for epoch in range(n_epochs):
        for datapoint in dataloader:
            imgs_a, imgs_b, a_R_b, a_t_b = [d.to(device) for d in datapoint]
            # coords_a, cs_a, heatmaps_a, cs_raw_a, coords_b, cs_b, heatmaps_b, cs_raw_b = model_1(imgs_a, imgs_b)
            a_t_b = torch.mul(a_t_b, 1 / 2.2857)
            coords_a, cs_a, heatmaps_a, cs_raw_a = model_1(imgs_a)
            coords_b, cs_b, heatmaps_b, cs_raw_b = model_1(imgs_b)

            B = imgs_a.shape[0]


            T = a_t_b.view(a_t_b.shape[0],1, a_t_b.shape[-1]).repeat(1,3,1)
            pos_loss = F.mse_loss((coords_b - coords_a), T)


            orn_loss = F.mse_loss((cs_b - cs_a), a_R_b)

            loss = pos_loss + orn_loss
            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())
            pbar.update()




train()