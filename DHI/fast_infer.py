# coding: utf-8
import argparse
from collections import defaultdict
import cv2
import imageio
import numpy as np
import os

import torch
import torch.nn as nn

import resnet
from timer import Timer
from utils import transformer as trans


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
    def __init__(self, args):
        exp_name = os.path.abspath(os.path.join(os.path.dirname("__file__"), os.path.pardir))
        self.result_name = "exp_fast"
        self.result_files = os.path.join(exp_name, self.result_name)
        if not os.path.exists(self.result_files):
            os.makedirs(self.result_files)

        self.model = resnet.resnet34()
        self.model.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Linear(512, 8)

        print(self.model)

        model_dict = self.model.state_dict()
        model_path = os.path.join(exp_name, 'models/freeze-mask-first-fintune.pth')
        checkpoint = torch.load(model_path, map_location='cpu').state_dict()
        # because the model is trained by multiple gpus, prefix module should be removed
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        self.model.load_state_dict(model_dict)

        self.model.cuda()
        self.model.eval()

        M_tensor = torch.tensor([[args.img_w/ 2.0, 0., args.img_w/ 2.0],
                                [0., args.img_h / 2.0, args.img_h / 2.0],
                                [0., 0., 1.]])
        M_tensor = M_tensor.cuda()

        self.M_tile = M_tensor.unsqueeze(0).expand(1, M_tensor.shape[-2], M_tensor.shape[-1])
        # Inverse of M
        M_tensor_inv = torch.inverse(M_tensor)
        self.M_tile_inv = M_tensor_inv.unsqueeze(0).expand(1, M_tensor_inv.shape[-2], M_tensor_inv.shape[-1])

        self.mean_I = np.reshape(np.array([118.93, 113.97, 102.60]), (1, 1, 3))
        self.std_I = np.reshape(np.array([69.85, 68.81, 72.45]), (1, 1, 3))

        self.patch_h = args.patch_size_h
        self.patch_w = args.patch_size_w
        self.WIDTH = args.img_w
        self.HEIGHT = args.img_h
        self.x_mesh, self.y_mesh = make_mesh(self.patch_w, self.patch_h)
        self.timers = defaultdict(Timer)
        
    def __call__(self, img_1, img_2):
        self.timers['all_time'].tic()
        self.timers['data'].tic()
        height, width = img_1.shape[:2]
        if height != self.HEIGHT or width != self.WIDTH:
            img_1 = cv2.resize(img_1, (self.WIDTH, self.HEIGHT))
        print_img_1_d = img_1.copy()
        img_1 = (img_1 - self.mean_I) / self.std_I
        img_1 = np.mean(img_1, axis=2, keepdims=True)
        img_1 = np.transpose(img_1, [2, 0, 1])

        if height != self.HEIGHT or width != self.WIDTH:
            img_2 = cv2.resize(img_2, (self.WIDTH, self.HEIGHT))
        print_img_2_d = img_2.copy()
        img_2 = (img_2 - self.mean_I) / self.std_I
        img_2 = np.mean(img_2, axis=2, keepdims=True)
        img_2 = np.transpose(img_2, [2, 0, 1])
        org_img = np.concatenate([img_1, img_2], axis=0)
        WIDTH = org_img.shape[2]
        HEIGHT = org_img.shape[1]

        x = 40  # patch should in the middle of full img when testing
        y = 23  # patch should in the middle of full img when testing
        input_tesnor = org_img[:, y: y + self.patch_h, x: x + self.patch_w]

        y_t_flat = np.reshape(self.y_mesh, [-1])
        x_t_flat = np.reshape(self.x_mesh, [-1])
        patch_indices = (y_t_flat + y) * WIDTH + (x_t_flat + x)

        top_left_point = (x, y)
        bottom_left_point = (x, y + self.patch_h)
        bottom_right_point = (self.patch_w + x, self.patch_h + y)
        top_right_point = (x + self.patch_w, y)
        four_points = [top_left_point, bottom_left_point, bottom_right_point, top_right_point]

        four_points = np.reshape(four_points, (-1))

        org_imges = torch.from_numpy(org_img[None]).float().cuda()
        input_tesnors = torch.from_numpy(input_tesnor[None]).float().cuda()
        patch_indices = torch.from_numpy(patch_indices[None]).float().cuda()
        h4p = torch.from_numpy(four_points[None]).float().cuda()

        print_img_1 = np.transpose(print_img_1_d, [2, 0, 1])
        print_img_1 = torch.from_numpy(print_img_1[None]).float().cuda()
        print_img_2 = np.transpose(print_img_2_d, [2, 0, 1])
        print_img_2 = torch.from_numpy(print_img_2[None]).float().cuda()
        self.timers['data'].toc()
        
        self.timers['model'].tic()
        batch_out = self.model(org_imges, input_tesnors, h4p, patch_indices)
        H_mat = batch_out['H_mat']
        self.timers['model'].toc()

        output_size = (self.HEIGHT, self.WIDTH)

        self.timers['post_process'].tic()
        H_mat = torch.matmul(torch.matmul(self.M_tile_inv, H_mat), self.M_tile)
        pred_full, _ = trans(print_img_1, H_mat, output_size)  # pred_full = warped imgA
        pred_full = pred_full.cpu().detach().numpy()[0, ...]
        pred_full = pred_full.astype(np.uint8)
        cv2.imwrite(os.path.join(self.result_files, "output.jpg"), pred_full)
        self.timers['post_process'].toc()

        # timers['make_gif'].tic()
        # pred_full = cv2.cvtColor(pred_full, cv2.COLOR_BGR2RGB)
        # print_img_1_d = cv2.cvtColor(print_img_1_d, cv2.COLOR_BGR2RGB)
        # print_img_2_d = cv2.cvtColor(print_img_2_d, cv2.COLOR_BGR2RGB)

        # input_list = [print_img_1_d, print_img_2_d]
        # output_list = [pred_full, print_img_2_d]
        # create_gif(input_list, os.path.join(self.result_files, "_input_[" + self.result_name + "].gif"))
        # create_gif(output_list, os.path.join(self.result_files, "_output_[" + self.result_name + "].gif"))
        # timers['make_gif'].toc()
        self.timers['all_time'].toc()

        for k, v in self.timers.items():
            if k != 'all_time':
                print(' | {}: {:.3f}s'.format(k, v.average_time))
        print(' ------| {}: {:.3f}s'.format('all_time', self.timers['all_time'].average_time))


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

    parser.add_argument('--model_name', type=str, default='resnet34')
    parser.add_argument('--pretrained', type=bool, default=False, help='Use pretrained waights?')
    parser.add_argument('--finetune', type=bool, default=True, help='Use pretrained waights?')

    print('<==================== Loading data ===================>\n')

    args = parser.parse_args()
    print(args)

    img_1 = cv2.imread('../images/00000238_10153.jpg')
    img_2 = cv2.imread('../images/00000238_10156.jpg')
    tt = Test(args)
    tt(img_1, img_2)
    # for i in range(1000):
    #     tt(img_1, img_2)