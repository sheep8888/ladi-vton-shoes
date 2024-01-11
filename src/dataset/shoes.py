import json
import os
import pickle
import random
import sys
from pathlib import Path
from typing import Tuple, Literal

PROJECT_ROOT = Path(__file__).absolute().parents[2].absolute()
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image, ImageDraw

from src.utils.posemap import get_coco_body25_mapping
from src.utils.posemap import kpoint_to_heatmap


class ShoesDataset(data.Dataset):
    def __init__(self,
                 dataroot_path: str,
                 phase: Literal['train', 'test'],
                 radius=5,
                 order: Literal['paired', 'unpaired'] = 'paired',
                 outputlist: Tuple[str] = ('c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'shape', 'pose_map',
                                           'parse_array', 'im_mask', 'inpaint_mask', 'parse_mask_total',
                                           'captions', 'category', 'warped_cloth', 'clip_cloth_features'),
                 size: Tuple[int, int] = (512, 384),
                 ):

        super(ShoesDataset, self).__init__()
        self.dataroot = dataroot_path
        self.phase = phase
        self.category = ('lower_body')
        self.outputlist = outputlist
        self.height = size[0]
        self.width = size[1]
        self.radius = radius
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.transform2D = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        self.order = order

        possible_outputs = ['c_name', 'im_name', 'cloth', 'image', 'im_cloth', 'shape', 'im_head', 'im_pose',
                            'pose_map', 'parse_array', 'dense_labels', 'dense_uv', 'skeleton',
                            'im_mask', 'inpaint_mask', 'parse_mask_total', 'captions',
                            'category', 'warped_cloth', 'clip_cloth_features']

        assert all(x in possible_outputs for x in outputlist)
        
        im_names = os.listdir(os.path.join(dataroot_path,self.phase,'ip'))
        im_names = [ i for i in im_names if int(i.split('.')[0]) > 1500]
        
        self.im_names = im_names
        self.c_name = im_names



    def __getitem__(self, index):
        im_name = self.im_names[index]
        c_name = im_name
        category = 'lower_body'
        dataroot =self.dataroot


        if "cloth" in self.outputlist:  # In-shop clothing image
            # Clothing image
            cloth = Image.open(os.path.join(dataroot, self.phase, 'ic', im_name))
            cloth = cloth.resize((self.width, self.height))
            cloth = self.transform(cloth)  # [-1,1]

        if "image" in self.outputlist:
            # Person image
            image = Image.open(os.path.join(dataroot, self.phase, 'ip', im_name))
            image = image.resize((self.width, self.height))
            image = self.transform(image)  # [-1,1]



        labels = {
            0: ['background', [0, 10]],  # 0 is background, 10 is neck
            1: ['hair', [1, 2]],  # 1 and 2 are hair
            2: ['face', [4, 13]],
            3: ['upper', [5, 6, 7]],
            4: ['bottom', [9, 12]],
            5: ['left_arm', [14]],
            6: ['right_arm', [15]],
            7: ['left_leg', [16]],
            8: ['right_leg', [17]],
            9: ['left_shoe', [18]],
            10: ['right_shoe', [19]],
            11: ['socks', [8]],
            12: ['noise', [3, 11]]
        }


        if "im_pose" in self.outputlist or "parser_mask" in self.outputlist or "im_mask" in self.outputlist or "parse_mask_total" in self.outputlist or "parse_array" in self.outputlist or "pose_map" in self.outputlist or "parse_array" in self.outputlist or "shape" in self.outputlist or "im_head" in self.outputlist:


            # Load pose points
            pose_name = im_name.replace('jpg','json')
            with open(os.path.join(dataroot, self.phase, 'jp', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]

                # rescale keypoints on the base of height and width
                pose_data[:, 0] = pose_data[:, 0] * (self.width / 768)
                pose_data[:, 1] = pose_data[:, 1] * (self.height / 1024)

            pose_mapping = get_coco_body25_mapping()

            # point_num = pose_data.shape[0]
            point_num = len(pose_mapping)

            pose_map = torch.zeros(point_num, self.height, self.width)
            r = self.radius * (self.height / 512.0)
            im_pose = Image.new('L', (self.width, self.height))
            pose_draw = ImageDraw.Draw(im_pose)
            neck = Image.new('L', (self.width, self.height))
            neck_draw = ImageDraw.Draw(neck)
            for i in range(point_num):
                one_map = Image.new('L', (self.width, self.height))
                draw = ImageDraw.Draw(one_map)

                point_x = np.multiply(pose_data[pose_mapping[i], 0], 1)
                point_y = np.multiply(pose_data[pose_mapping[i], 1], 1)

                if point_x > 1 and point_y > 1:
                    draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    pose_draw.rectangle((point_x - r, point_y - r, point_x + r, point_y + r), 'white', 'white')
                    if i == 2 or i == 5:
                        neck_draw.ellipse((point_x - r * 4, point_y - r * 4, point_x + r * 4, point_y + r * 4), 'white',
                                          'white')
                one_map = self.transform2D(one_map)
                pose_map[i] = one_map[0]

            d = []

            for idx in range(point_num):
                ux = pose_data[pose_mapping[idx], 0]  # / (192)
                uy = (pose_data[pose_mapping[idx], 1])  # / (256)

                # scale posemap points
                px = ux  # * self.width
                py = uy  # * self.height

                d.append(kpoint_to_heatmap(np.array([px, py]), (self.height, self.width), 9))

            pose_map = torch.stack(d)
        tf = transforms.ToTensor()
        inpaint_mask = Image.open(os.path.join(dataroot, self.phase, 'im', im_name))
        inpaint_mask = inpaint_mask.resize((self.width, self.height))
        inpaint_mask = tf(inpaint_mask)
        im_mask = Image.open(os.path.join(dataroot, self.phase, 'ia', im_name))
        im_mask = im_mask.resize((self.width, self.height))
        im_mask = tf(im_mask)
        if "im_cloth" in self.outputlist:
            im_cloth = image * inpaint_mask
        if "warped_cloth" in self.outputlist:  # Precomputed warped clothing image
            # warped_cloth = Image.open(
            #     os.path.join(os.path.join(dataroot, self.phase, 'warped_cloth', im_name.split('.')[0]+'_'+im_name)))
            warped_cloth = inpaint_mask * image
            # warped_cloth = warped_cloth.resize((self.width, self.height))
            # warped_cloth = self.transform(warped_cloth)  # [-1,1]


        result = {}
        for k in self.outputlist:
            result[k] = vars()[k]

        return result

    def __len__(self):
        return len(self.im_names)
