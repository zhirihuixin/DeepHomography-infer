# coding: utf-8
import argparse
from collections import defaultdict
import cv2
import imageio
import math
import numpy as np
import os

import torch
import torch.nn as nn
from torch2trt import torch2trt
from torch2trt import TRTModule

import resnet
from homography_model import ShareFeatureModel, GenMaskModel
from timer import Timer
from utils import DLT_solve, transformer


class AffineChannel2d(nn.Module):
    """ A simple channel-wise affine transformation operation """
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.weight.data.uniform_()
        self.bias.data.zero_()

    def forward(self, x):
        return x * self.weight.view(1, self.num_features, 1, 1) + \
            self.bias.view(1, self.num_features, 1, 1)


def getPatchFromFullimg(patch_size_h, patch_size_w, patchIndices, batch_indices_tensor, img_full):
    num_batch, num_channels, height, width = img_full.size()
    warped_images_flat = img_full.reshape(-1)
    patch_indices_flat = patchIndices.reshape(-1)

    pixel_indices = patch_indices_flat.long() + batch_indices_tensor
    mask_patch = torch.gather(warped_images_flat, 0, pixel_indices)
    mask_patch = mask_patch.reshape([num_batch, 1, patch_size_h, patch_size_w])

    return mask_patch


def normMask(mask, strenth = 0.5):
    """
    :return: to attention more region

    """
    batch_size, c_m, c_h, c_w = mask.size()
    max_value = mask.reshape(batch_size, -1).max(1)[0]
    max_value = max_value.reshape(batch_size, 1, 1, 1)
    mask = mask/(max_value*strenth)
    mask = torch.clamp(mask, 0, 1)

    return mask


def create_gif(image_list, gif_name, duration=0.35):
    frames = []
    for image_name in image_list:
        frames.append(image_name)
    imageio.mimsave(gif_name, frames, 'GIF', duration=0.5)
    return


def make_mesh(patch_w,patch_h):
    x_flat = np.arange(0,patch_w)
    x_flat = x_flat[np.newaxis,:]
    y_one = np.ones(patch_h)
    y_one = y_one[:,np.newaxis]
    x_mesh = np.matmul(y_one , x_flat)

    y_flat = np.arange(0,patch_h)
    y_flat = y_flat[:,np.newaxis]
    x_one = np.ones(patch_w)
    x_one = x_one[np.newaxis,:]
    y_mesh = np.matmul(y_flat,x_one)
    return x_mesh,y_mesh


class Test(object):
    def __init__(self, args, img_2):
        exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
        self.result_name = "exp_fast"
        self.result_files = os.path.join(exp_name, self.result_name)
        if not os.path.exists(self.result_files):
            os.makedirs(self.result_files)

        self.patch_h = args.patch_size_h
        self.patch_w = args.patch_size_w
        self.WIDTH = args.img_w
        self.HEIGHT = args.img_h
        self.half_size = args.half_size
        self.use_trt = args.use_trt
        self.save_img = args.save_img

        model_path = os.path.join(exp_name, 'models/freeze-mask-first-fintune.pth')
        self.model, self.sfmodel, self.gmmodel = self.initialize_model(model_path)

        M_tensor = torch.tensor([[args.img_w/ 2.0, 0., args.img_w/ 2.0],
                                [0., args.img_h / 2.0, args.img_h / 2.0],
                                [0., 0., 1.]])
        M_tensor = M_tensor.cuda()

        self.M_tile = M_tensor.unsqueeze(0).expand(1, M_tensor.shape[-2], M_tensor.shape[-1])
        # Inverse of M
        M_tensor_inv = torch.inverse(M_tensor)
        self.M_tile_inv = M_tensor_inv.unsqueeze(0).expand(1, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])

        self.preproc = AffineChannel2d(3)
        self.preproc.weight.data = torch.from_numpy(1. / np.array([69.85, 68.81, 72.45])).float()
        self.preproc.bias.data = torch.from_numpy(-1. * np.array([118.93, 113.97, 102.60]) /
                                                  np.array([69.85, 68.81, 72.45])).float()
        self.preproc.cuda().eval()

        self.x_mesh, self.y_mesh = make_mesh(self.patch_w, self.patch_h)
        self.timers = defaultdict(Timer)
        self.target_img(img_2)

    def target_img(self, img_2):
        if self.half_size:
            print_img_2_d = img_2.copy()
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))
        else:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))
            print_img_2_d = img_2.copy()

        img_2 = torch.from_numpy(img_2.transpose((2, 0, 1))).cuda().float()[None]
        img_2 = self.preproc(img_2)
        img_2 = torch.mean(img_2, dim=1, keepdim=True)

        print_img_2 = np.transpose(print_img_2_d, [2, 0, 1])
        print_img_2 = torch.from_numpy(print_img_2[None]).float().cuda()

        x = math.ceil((self.WIDTH - self.patch_w) / 2)
        y = math.ceil((self.HEIGHT - self.patch_h) / 2)
        input_tesnor_2 = img_2[:, :,  y: y + self.patch_h, x: x + self.patch_w]

        y_t_flat = np.reshape(self.y_mesh, [-1])
        x_t_flat = np.reshape(self.x_mesh, [-1])
        patch_indices = (y_t_flat + y) * self.WIDTH + (x_t_flat + x)

        top_left_point = (x, y)
        bottom_left_point = (x, y + self.patch_h)
        bottom_right_point = (self.patch_w + x, self.patch_h + y)
        top_right_point = (x + self.patch_w, y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        four_points = np.reshape(four_points, (-1))
        self.patch_indices = torch.from_numpy(patch_indices[None]).float().cuda()
        self.h4p = torch.from_numpy(four_points[None]).float().cuda()

        y_t = torch.arange(0, self.WIDTH * self.HEIGHT,
                           self.WIDTH * self.HEIGHT)
        batch_indices_tensor = y_t.unsqueeze(1).expand(y_t.shape[0], self.patch_h * self.patch_w).reshape(-1)
        self.batch_indices_tensor = batch_indices_tensor.cuda()

        mask_I2_full = self.gmmodel(img_2)
        mask_I2 = getPatchFromFullimg(
            self.patch_h, self.patch_w, self.patch_indices, self.batch_indices_tensor, mask_I2_full
        )
        mask_I2 = normMask(mask_I2)
        patch_2 = self.sfmodel(input_tesnor_2)
        self.patch_2_res = torch.mul(patch_2, mask_I2)

        self.print_img_2_d = cv2.cvtColor(print_img_2_d, cv2.COLOR_BGR2RGB)
        
    def __call__(self, img_1):
        torch.cuda.synchronize()
        self.timers['all_time'].tic()
        self.timers['data'].tic()
        if self.half_size:
            print_img_1_d = img_1.copy()
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))
        else:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))
            print_img_1_d = img_1.copy()

        img_1 = torch.from_numpy(img_1.transpose((2, 0, 1))).cuda().float()[None]
        img_1 = self.preproc(img_1)
        img_1 = torch.mean(img_1, dim=1, keepdim=True)

        print_img_1 = np.transpose(print_img_1_d, [2, 0, 1])
        print_img_1 = torch.from_numpy(print_img_1).cuda().float()[None]

        x = math.ceil((self.WIDTH - self.patch_w) / 2)
        y = math.ceil((self.HEIGHT - self.patch_h) / 2)
        input_tesnor_1 = img_1[:, :,  y: y + self.patch_h, x: x + self.patch_w]

        torch.cuda.synchronize()
        self.timers['data'].toc()
        self.timers['model'].tic()
        self.timers['gm_sf'].tic()

        mask_I1_full = self.gmmodel(img_1)
        
        mask_I1 = getPatchFromFullimg(
            self.patch_h, self.patch_w, self.patch_indices, self.batch_indices_tensor, mask_I1_full
        )
        mask_I1 = normMask(mask_I1)
        patch_1 = self.sfmodel(input_tesnor_1)

        patch_1_res = torch.mul(patch_1, mask_I1)
        x = torch.cat((patch_1_res, self.patch_2_res), dim=1)

        torch.cuda.synchronize()
        self.timers['gm_sf'].toc()
        self.timers['res34'].tic()
        x = self.model(x)
        torch.cuda.synchronize()
        self.timers['res34'].toc()
        self.timers['DLT_solve'].tic()
        H_mat = DLT_solve(self.h4p, x).squeeze(1)
        # print(H_mat)
        torch.cuda.synchronize()
        self.timers['DLT_solve'].toc()
        self.timers['model'].toc()
        if self.half_size:
            w = self.WIDTH * 2
            h = self.HEIGHT * 2
        else:
            w = self.WIDTH
            h = self.HEIGHT
        torch.cuda.synchronize()
        self.timers['post_process'].tic()
        # pred_full = cv2.warpPerspective(
        #         print_img_1_d, H_mat.cpu().detach().numpy()[0], 
        #         (w, h)
        #     )
        H_mat = torch.matmul(torch.matmul(self.M_tile_inv, H_mat), self.M_tile)
        # print(H_mat)
        pred_full, _ = transformer(print_img_1, H_mat, (h, w))  # pred_full = warped imgA
        pred_full = pred_full[0].type(torch.uint8).cpu().numpy()
        pred_full = pred_full.astype(np.uint8)
        torch.cuda.synchronize()
        self.timers['post_process'].toc()
        if self.save_img:
            self.timers['save_img'].tic()
            cv2.imwrite(os.path.join(self.result_files, "output.jpg"), pred_full)
            torch.cuda.synchronize()
            self.timers['save_img'].toc()

            self.timers['make_gif'].tic()
            pred_full = cv2.cvtColor(pred_full, cv2.COLOR_BGR2RGB)
            print_img_1_d = cv2.cvtColor(print_img_1_d, cv2.COLOR_BGR2RGB)

            input_list = [print_img_1_d, self.print_img_2_d]
            output_list = [pred_full, self.print_img_2_d]
            change_list = [pred_full, print_img_1_d]
            create_gif(input_list, os.path.join(self.result_files, "input.gif"))
            create_gif(output_list, os.path.join(self.result_files, "output.gif"))
            create_gif(change_list, os.path.join(self.result_files, "change.gif"))
            self.timers['make_gif'].toc()

        torch.cuda.synchronize()
        self.timers['all_time'].toc()
        for k, v in self.timers.items():
            if k != 'all_time':
                print(' | {}: {:.3f}s'.format(k, v.average_time))
        print(' ------| {}: {:.3f}s'.format('all_time', self.timers['all_time'].average_time))

    def initialize_model(self, model_path):
        model = resnet.resnet34()
        model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(512, 8)
        print(model)
        model_dict = model.state_dict()

        sfmodel = ShareFeatureModel()
        sfmodel_dict = sfmodel.state_dict()

        gmmodel = GenMaskModel()
        gmmodel_dict = gmmodel.state_dict()

        checkpoint = torch.load(model_path, map_location='cpu').state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            if 'ShareFeature' in k:
                sfmodel_dict[k.replace('module.', '')] = checkpoint[k]
            elif 'genMask' in k:
                gmmodel_dict[k.replace('module.', '')] = checkpoint[k]
            else:
                model_dict[k.replace('module.', '')] = checkpoint[k]
        model.load_state_dict(model_dict)
        model.cuda().eval()
        sfmodel.load_state_dict(sfmodel_dict)
        sfmodel.cuda().eval()
        gmmodel.load_state_dict(gmmodel_dict)
        gmmodel.cuda().eval()

        if self.use_trt:
            x = torch.ones((1, 1, self.HEIGHT, self.WIDTH)).cuda()
            # convert to TensorRT feeding sample data as input
            gmmodel_trt = torch2trt(gmmodel, [x], fp16_mode=True)
            x = torch.ones((1, 1, self.patch_h, self.patch_w)).cuda()
            # convert to TensorRT feeding sample data as input
            sfmodel_trt = torch2trt(sfmodel, [x], fp16_mode=True)
            x = torch.ones((1, 2, self.patch_h, self.patch_w)).cuda()
            # convert to TensorRT feeding sample data as input
            model_trt = torch2trt(model, [x], fp16_mode=True)
            return model_trt, sfmodel_trt, gmmodel_trt
        else:
            return model, sfmodel, gmmodel


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=int, default=4, help='Number of splits')
    parser.add_argument('--cpus', type=int, default=10, help='Number of cpus')

    parser.add_argument('--img_w', type=int, default=640)
    parser.add_argument('--img_h', type=int, default=360)
    parser.add_argument('--patch_size_h', type=int, default=315)
    parser.add_argument('--patch_size_w', type=int, default=560)

    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-9, help='learning rate')

    parser.add_argument('--half_size', type=bool, default=False, help='Use half size inference?')
    parser.add_argument('--use_trt', type=bool, default=False, help='Use tensorRT?')
    parser.add_argument('--save_img', type=bool, default=True, help='save png and gif?')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)

    # img_1 = cv2.imread('../images/00000238_10153.jpg')
    # img_2 = cv2.imread('../images/00000238_10156.jpg')

    img_1 = cv2.imread('../images/0000026_10028.jpg')
    img_2 = cv2.imread('../images/0000026_10001.jpg')

    args.img_w = 2448 // 1
    args.img_h = 2048 // 1
    args.patch_size_w = args.img_w * 7 // 8
    args.patch_size_h = args.img_h * 7 // 8
    args.half_size = False
    args.use_trt = True
    args.save_img = False

    tt = Test(args, img_2)
    tt(img_1)
    for i in range(1000):
        tt(img_1)