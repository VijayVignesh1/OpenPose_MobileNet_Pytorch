import argparse
import cv2
import json
import math
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch
import os

from datasets.coco import ValDataset
from models.with_mobilenet import PoseEstimationWithMobileNet
from modules.keypoints import extract_keypoints
from modules.load_state import load_state
import config as args
import glob
from tqdm import tqdm

# Normalize the image values to the range [0-1]
def normalize(img, img_mean, img_scale):
    img = np.array(img, dtype=np.float32)
    img = (img - img_mean) * img_scale
    return img

# Gets the images as input and applies the model to it
# Returns the heatmap which gives the probability of the keypoint in the images
def infer(net, img, scales, base_height, stride, pad_value=(0, 0, 0), img_mean=(128, 128, 128), img_scale=1/256):
    normed_img = normalize(img, img_mean, img_scale)
    height, width, _ = normed_img.shape
    scales_ratios = [scale * base_height / float(height) for scale in scales]
    avg_heatmaps = np.zeros((height, width, args.num_keypoints), dtype=np.float32)

    for ratio in scales_ratios:
        tensor_img = torch.from_numpy(normed_img).permute(2, 0, 1).unsqueeze(0).float().cuda()
        stages_output = net(tensor_img)

        stage2_heatmaps = stages_output[-1]
        heatmaps = np.transpose(stage2_heatmaps.squeeze().cpu().data.numpy(), (1, 2, 0))
        heatmaps = cv2.resize(heatmaps, (0, 0), fx=stride, fy=stride, interpolation=cv2.INTER_CUBIC)
        heatmaps = cv2.resize(heatmaps, (width, height), interpolation=cv2.INTER_CUBIC)
        avg_heatmaps = avg_heatmaps + heatmaps / len(scales_ratios)

    return avg_heatmaps

# Evaulate the model on given images
# Visualizes the output if visualize = True
def evaluate(val_file_name, output_name, images_folder, net, num_iter, multiscale=False, visualize=True):
    net = net.cuda().eval()
    base_height = 368
    scales = [1]
    if multiscale:
        scales = [0.5, 1.0, 1.5, 2.0]
    stride = 8

    dataset = ValDataset(val_file_name, images_folder)
    coco_result = []
    for sample in dataset:
        file_name = sample['file_name']
        img = sample['img']

        avg_heatmaps = infer(net, img, scales, base_height, stride)
        total_keypoints_num = 0
        all_keypoints_by_type = []
        for kpt_idx in range(args.num_keypoints):  # 19th for bg
            total_keypoints_num += extract_keypoints(avg_heatmaps[:, :, kpt_idx], all_keypoints_by_type, total_keypoints_num)

        points=[]
        for i in all_keypoints_by_type:
            if i==[]:
                continue
            elif len(i)==1:
                points.append((i[0][0],i[0][1]))
            else:
                max_x=0
                max_y=0
                max_score=0
                for j in i:
                    if j[3]>max_score:
                        max_score=j[3]
                        max_x=j[0]
                        max_y=j[1]
                points.append((max_x,max_y))

        if visualize:
            for i in points:
                cv2.circle(img,i,3,(255,0,255),-1)
            output_name1=os.path.join(images_folder,output_name)+str(num_iter)+".jpg"
            cv2.imwrite(output_name1,img)
            return


if __name__ == '__main__':
    net = PoseEstimationWithMobileNet()
    checkpoint = torch.load(args.checkpoint_path)
    load_state(net, checkpoint)
    files=glob.glob(args.validation_folder+"/*.jpg")
    count=0
    if not os.path.exists(args.validation_output_folder):
        os.makedirs(args.validation_output_folder)    
    for i in tqdm(files):
        file_name=os.path.split(i)[1] 
        evaluate(i, file_name, args.validation_output_folder, net, "", args.multiscale, args.visualize)
