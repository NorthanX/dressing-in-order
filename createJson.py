import torch
from models.dior_model import DIORModel
import os, json
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from utils import pose_utils


def load_pose_from_json(pose_json, target_size=(512, 352), orig_size=(384, 384)):
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
        return torch.zeros((18, a, b))
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
    print(pose)
    pose = np.transpose(pose, (2, 0, 1))
    pose = torch.Tensor(pose)
    return pose[:18]


pose = load_pose_from_json("./json/fashionMENTees_Tanksid0000595506_1front_keypoints.json")
print(pose)

