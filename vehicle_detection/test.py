import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
import os
import json
from detectron2.structures import BoxMode

# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate

import argparse

parser = argparse.ArgumentParser(description='Arguments for detectron')
parser.add_argument('--scene',type=str, default = 'tiny_foggy', help='data scene number')
parser.add_argument('--folder',type=str, default ='Navtech_Cartesian', help='front data for True and rear data for False')
parser.add_argument('--total',type=int, default =20, help='Total images per scene')
parser.add_argument('--ckpt',type=str, default ='default', help='checkpoint for eval')
args = parser.parse_args()

debug = False
root_dir = '../data/radiate/'

def gen_boundingbox(bbox, angle):
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

        min_x = np.min(points[0, :])
        min_y = np.min(points[1, :])
        max_x = np.max(points[0, :])
        max_y = np.max(points[1, :])

        return min_x, min_y, max_x, max_y


def get_radar_dicts(folders,curr_dir):
        category_to_id={'pedestrian':0, 'group_of_pedestrians':1, 'bus':2, 'car':3,'van':4,'truck':5,'motorbike':6,'bicycle':7}
        dataset_dicts = []
        idd = 0
        folder_size = len(folders)
        for folder in folders:
            radar_folder = os.path.join(root_dir, folder, curr_dir)
            annotation_path = os.path.join(root_dir,
                                           folder, 'annotations', 'annotations.json')
            with open(annotation_path, 'r') as f_annotation:
                annotation = json.load(f_annotation)

            radar_files = os.listdir(radar_folder)
            radar_files.sort()
            
            if args.scene=='ALL':
                radar_cart_folder = os.path.join(root_dir, folder, 'Navtech_Cartesian')
                total_cart_imgs = len(os.listdir(radar_cart_folder))
                total_images = len(radar_files) if len(radar_files) < total_cart_imgs else total_cart_imgs
            else:
                total_images = args.total
            for frame_number in range(total_images):
            #for frame_number in range(len(radar_files)):
                record = {}
                objs = []
                bb_created = False
                idd += 1
                filename = os.path.join(
                    radar_folder, radar_files[frame_number])
                #print(filename)

                if (not os.path.isfile(filename)):
                    print(filename)
                    continue
                record["file_name"] = filename
                record["image_id"] = idd
                record["height"] = 1152
                record["width"] = 1152

                if debug:
                    print("folder:", filename, frame_number)
                for object in annotation:
                    if (object['bboxes'][frame_number]):
                            class_obj = object['class_name']
                            if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):
                                bbox = object['bboxes'][frame_number]['position']
                                angle = object['bboxes'][frame_number]['rotation']
                                bb_created = True
                                if cfg.MODEL.PROPOSAL_GENERATOR.NAME == "RRPN":
                                    cx = bbox[0] + bbox[2] / 2
                                    cy = bbox[1] + bbox[3] / 2
                                    wid = bbox[2]
                                    hei = bbox[3]
                                    obj = {
                                        "bbox": [cx, cy, wid, hei, angle],
                                        "bbox_mode": BoxMode.XYWHA_ABS,
                                        "category_id": 0,
                                        "iscrowd": 0
                                          }
                                else:
                                    xmin, ymin, xmax, ymax = gen_boundingbox(
                                        bbox, angle)
                                    #xmin, ymin, xmax, ymax = bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[0]
                                    obj = {
                                        "bbox": [xmin, ymin, xmax, ymax],
                                        "bbox_mode": BoxMode.XYXY_ABS,
                                        "category_id": 0,
                                        "iscrowd": 0
                                          }

                                objs.append(obj)
                if bb_created:
                    record["annotations"] = objs
                    dataset_dicts.append(record)
        return dataset_dicts







network = 'faster_rcnn_R_50_FPN_3x'   #########
#network = 'faster_rcnn_R_101_FPN_3x'
setting = 'good_and_bad_weather_radar'

# time (s) to retrieve next frame
dt = 0.25


cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
#cfg.MODEL.DEVICE = 'cpu'

if args.ckpt == 'default':
    cfg.MODEL.WEIGHTS = os.path.join('weights',  network +'_' + setting + '.pth')
else:
    cfg.MODEL.WEIGHTS = args.ckpt
print("Inference model:", cfg.MODEL.WEIGHTS)
#cfg.MODEL.WEIGHTS = 'train_results_latest/faster_rcnn_R_50_FPN_3x_good_and_bad_weather/model_final.pth'
#cfg.MODEL.WEIGHTS = 'train_results/faster_rcnn_trained/model_final.pth'
#cfg.MODEL.WEIGHTS = 'train_results_polar/fine_tune_100iter_city_1_3_city_7_0_rad_200/model_final.pth' ###
#cfg.MODEL.WEIGHTS = 'train_results/finetune_rad_000001lr_80iter_city_1_3_city_7_0/model_final.pth'
#cfg.MODEL.WEIGHTS = 'train_results/finetune_rad_100lr_200iter_30_rad_info/model_final.pth'

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)

cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2

#cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 #0.05 #0.05   #############

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)


root_dir = '../data/radiate/'
folders_test = []
for curr_dir in os.listdir(root_dir):
    with open(os.path.join(root_dir, curr_dir, 'meta.json')) as f:
        meta = json.load(f)
    if meta["set"] == "test":
        folders_test.append(curr_dir)
print(folders_test)


import torch
state_dict = torch.load(cfg.MODEL.WEIGHTS)
params = 0
for key in state_dict['model']:
	if 'weight' in key:
		params += state_dict['model'][key].numel()

print(params)

#'''
if args.scene == 'ALL':
    folders_test = folders_test
    folders_test.remove('tiny_foggy')
elif args.scene== 'all':
    folders_test = ['city_3_7','fog_6_0','snow_1_0','night_1_4','motorway_2_2'] #tiny_foggy
    #folders_test = ['rain_3_0','night_1_5','rain_4_0']
else:
    folders_test=[args.scene]
#'''

print("test folders:", folders_test)
dataset_test_name = 'test'

if args.scene == 'ALL' and args.folder=='Navtech_Cartesian':# or args.total >20:
    curr_dir = args.folder
else:
    curr_dir = args.folder + '/radar-cart-img'   ##############
print("curr_dir:", curr_dir)
DatasetCatalog.register(dataset_test_name,
                            lambda: get_radar_dicts(folders_test,curr_dir))
MetadataCatalog.get(dataset_test_name).set(thing_classes=["vehicle"])
dataset_test_name = 'test'
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator(dataset_test_name,cfg,True, output_dir="./output")
val_loader = build_detection_test_loader(cfg, dataset_test_name)
print(inference_on_dataset(predictor.model, val_loader, evaluator))

