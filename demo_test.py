import torch
from models.dior_model import DIORModel
import os, json
import torch.nn.functional as F
from PIL import Image
import pylab

import matplotlib.pyplot as plt
import numpy as np

# dataroot = '/shared/rsaas/aiyucui2/inshop/fashion_yifang'
dataroot = 'dataroot'
exp_name = 'DIOR_64'  # DIORv1_64
epoch = 'latest'
netG = 'dior'  # diorv1
ngf = 64


## this is a dummy "argparse"
class Opt:
    def __init__(self):
        pass


if True:
    opt = Opt()
    opt.dataroot = dataroot
    opt.isTrain = False
    opt.phase = 'test'
    opt.n_human_parts = 8
    opt.n_kpts = 18
    opt.style_nc = 64
    opt.n_style_blocks = 4
    opt.netG = netG
    opt.netE = 'adgan'
    opt.ngf = ngf
    opt.norm_type = 'instance'
    opt.relu_type = 'leakyrelu'
    opt.init_type = 'orthogonal'
    opt.init_gain = 0.02
    opt.gpu_ids = [0]
    opt.frozen_flownet = True
    opt.random_rate = 1
    opt.perturb = False
    opt.warmup = False
    opt.name = exp_name
    opt.vgg_path = ''
    opt.flownet_path = 'pretrained_models/latest_net_Flow.pth'
    opt.checkpoints_dir = 'checkpoints'
    opt.frozen_enc = True
    opt.load_iter = 0
    opt.epoch = epoch
    opt.verbose = False

# create model
model = DIORModel(opt)
model.setup(opt)

# 测试matplotlib

from pandas import Series, DataFrame
import matplotlib

matplotlib.use('TkAgg')
# x = np.arange(0, 10, step=1)
# s1 = Series(x, index=list('abcdefghij'))
# s2 = Series(x ** 2, index=s1.index)
#
# # 索引和值一起设置
# plt.plot(x, x * 2, x, x * 3, x, x)
#
# plt.show()


from utils import pose_utils

# load data
from datasets.deepfashion_datasets import DFVisualDataset

Dataset = DFVisualDataset
ds = Dataset(dataroot=dataroot, dim=(256, 176), n_human_part=8)

# preload a set of pre-selected models defined in "standard_test_anns.txt" for quick visualizations
inputs = dict()
for attr in ds.attr_keys:
    inputs[attr] = ds.get_attr_visual_input(attr)


# define some tool functions for I/O
def load_img(pid, ds):
    if isinstance(pid, str):  # load pose from scratch
        return None, None, load_pose_from_json(pid)
    if len(pid[0]) < 10:  # load pre-selected models
        person = inputs[pid[0]]
        # print(person)
        person = (i.cuda() for i in person)
        pimg, parse, to_pose = person
        pimg, parse, to_pose = pimg[pid[1]], parse[pid[1]], to_pose[pid[1]]
    else:  # load model from scratch
        person = ds.get_inputs_by_key(pid)
        person = (i.cuda() for i in person)
        pimg, parse, to_pose = person
    return pimg.squeeze(), parse.squeeze(), to_pose.squeeze()


def load_pose_from_json(pose_json, target_size=(256, 176), orig_size=(384, 384)):
    '''
        This function converts the OpenPose detected key points (in .json file) to the desired heatmap.
        input:
        - pose_json (str): the file_path of the OpenPose detection in .json.
        - target_size (tuple): the size of output heatmap
        - orig_size (tuple): the size of original image that is used for OpenPose to detect the key points.
        Output:
        - heatmap (torch.Tensor) : the heatmap in size 18xHxW as specified by target_size
        '''
    with open(pose_json, 'r') as f:
        anno = json.load(f)
    if len(anno['people']) < 1:
        a, b = target_size
        return torch.zero((18, a, b))
    anno = list(anno['people'][0]['pose_keypoints_2d'])
    x = np.array(anno[1::3])
    y = np.array(anno[::3])

    x[8:-1] = x[9:]
    y = np.array(anno[::3])
    y[8:-1] = y[9:]
    x[x == 0] = -1
    y[y == 0] = -1
    coord = np.concatenate([x[:, None], y[:, None]], -1)
    pose = pose_utils.cords_to_map(coord, target_size, orig_size)
    pose = np.transpose(pose, (2, 0, 1))
    pose = torch.Tensor(pose)
    return pose[:18]


def plot_img(pimg=[], gimgs=[], oimgs=[], gen_img=[], pose=None):
    if type(pose) != type(None):
        import utils.pose_utils as pose_utils
        print(pose.size())
        kpt = pose_utils.draw_pose_from_map(pose.cpu().numpy().transpose(1, 2, 0), radius=6)
        kpt = kpt[0]
    if not isinstance(pimg, list):
        pimg = [pimg]
    if not isinstance(gen_img, list):
        gen_img = [gen_img]
    out = pimg + gimgs + oimgs + gen_img
    if out:
        out = torch.cat(out, 2).float().cpu().detach().numpy()
        out = (out + 1) / 2  # denormalize
        out = np.transpose(out, [1, 2, 0])

        if type(pose) != type(None):
            out = np.concatenate((kpt, out), 1)
    else:
        out = kpt
    fig = plt.figure(figsize=(6, 4), dpi=100, facecolor='w', edgecolor='k')
    plt.axis('off')
    plt.imshow(out)
    plt.show()


# define dressing-in-order function (the pipeline)
def dress_in_order(model, pid, pose_id=None, gids=[], ogids=[], order=[5, 1, 3, 2], perturb=False):
    PID = [0, 4, 6, 7]
    GID = [2, 5, 1, 3]
    # encode person
    pimg, parse, from_pose = load_img(pid, ds)
    if perturb:
        pimg = DIORModel.perturb_images(pimg[None])[0]
    if not pose_id:
        to_pose = from_pose
    else:
        to_img, _, to_pose = load_img(pose_id, ds)
    psegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None], PID)

    # encode base garments
    gsegs = model.encode_attr(pimg[None], parse[None], from_pose[None], to_pose[None])

    # swap base garment if any
    gimgs = []
    for gid in gids:
        _, _, k = gid
        gimg, gparse, pose = load_img(gid, ds)
        seg = model.encode_single_attr(gimg[None], gparse[None], pose[None], to_pose[None], i=gid[2])
        gsegs[gid[2]] = seg
        gimgs += [gimg * (gparse == gid[2]).to(torch.float32)]

    # # encode garment (overlay)
    # garments = []
    # over_gsegs = []
    # oimgs = []
    # for gid in ogids:
    #     oimg, oparse, pose = load_img(gid, ds)
    #     oimgs += [oimg * (oparse == gid[2]).to(torch.float32)]
    #     seg = model.encode_single_attr(oimg[None], oparse[None], pose[None], to_pose[None], i=gid[2])
    #     over_gsegs += [seg]

    # gsegs = [gsegs[i] for i in order] + over_gsegs
    gen_img = model.netG(to_pose[None], psegs, gsegs)

    return pimg, gimgs, oimgs, gen_img[0], to_pose


## Try-On: Tucking in

# pid = ("pattern", 3, None)  # load the 3-rd person from "pattern" group, NONE (no) garment is interested
pid = ("4-model.jpg", "./json/4-model_keypoints.json", None)
gids = [
    ("1-testClothWomen1.jpg", "./json/1-testClothWomen1_keypoints.json", 5), # load the 0-th person from "plaid" group, garment #5 (top) is interested
    ("1-testClothWomen1.jpg", "./json/1-testClothWomen1_keypoints.json", 1),
    # ("4-testbymyown.jpg", "./json/4-testbymyown_keypoints.json",1),  # load the 3-rd person from "pattern" group, garment #1 (bottom) is interested
]
# ("plaid",0,5)
# "./json/1_keypoints.json"

from torchvision import transforms
# not tuckin (dressing order: hair, bottom, top)
pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, gids=gids, order=[2, 1, 5])

plot_img(pimg, gimgs, gen_img=gen_img, pose=pose)

# image = image.squeeze(0)
unloader =transforms.ToPILImage()
# out = torch.cat(image, 2).float().cpu().detach().numpy()
out = gen_img.cpu().clone()
out = (out + 1) / 2  # denormalize
# out = np.transpose(out, [1, 2, 0])
image = unloader(out)
image.save('random.jpg')


# pid = ("fashionWOMENBlouses_Shirtsid0000637003_1front.jpg", None, None)  # load person from the file
#
# ogids = [("1-testClothMen5.jpg", "./json/1-testClothMen5_keypoints.json", 3)]
# gids = [
#     ("gfla",2,5),
#     ("strip",3,1),
#        ]
#
# pimg, gimgs, oimgs, gen_img, pose = dress_in_order(model, pid, ogids=ogids)
# plot_img(pimg, gimgs, oimgs, gen_img, pose)