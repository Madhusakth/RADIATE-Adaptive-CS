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
from skimage import io
# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate

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


def get_radar_dicts(folders):
        dataset_dicts = []
        idd = 0
        folder_size = len(folders)
        for folder in folders:
            radar_folder = os.path.join(root_dir, folder,'Navtech_Cartesian_20')
            annotation_path = os.path.join(root_dir,
                                           folder, 'annotations', 'annotations.json')
            with open(annotation_path, 'r') as f_annotation:
                annotation = json.load(f_annotation)

            radar_files = os.listdir(radar_folder)
            radar_files.sort()
            for frame_number in range(len(radar_files)):
                record = {}
                objs = []
                bb_created = False
                idd += 1
                filename = os.path.join(
                    radar_folder, radar_files[frame_number])

                if (not os.path.isfile(filename)):
                    print(filename)
                    continue
                record["file_name"] = filename
                record["image_id"] = idd
                record["height"] = 1152
                record["width"] = 1152


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






# path to the sequence
root_path = '../data/radiate/'
sequence_name = 'city_3_7'#'fog_6_0' #'snow_1_0' #'tiny_foggy' night_1_4 motorway_2_2
radar_path = '20-polar-kalman-10-multi-20-anchor-40-custom-test-8/radar-cart-img' #'20-final-rad-info-polar-kalman-3/radar-cart-img' #'Navtech_Cartesian_20' #'final-rad-info' #'reconstruct/reshaped' #'Navtech_Cartesian'

network = 'faster_rcnn_R_50_FPN_3x'
setting = 'good_and_bad_weather_radar'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/config/config.yaml',reconst_path = radar_path)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
#cfg.MODEL.DEVICE = 'cpu'
#cfg.MODEL.WEIGHTS = os.path.join('weights',  network +'_' + setting + '.pth')
cfg.MODEL.WEIGHTS = 'train_results_latest/faster_rcnn_R_50_FPN_3x_good_and_bad_weather/model_final.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

saveDir = root_path+ sequence_name + '/' + radar_path + '_annotated_nw_orig'
if not os.path.isdir(saveDir):
    os.mkdir(saveDir)

multi_class= False
category_to_id={1:'pedestrian', 2:'group_of_pedestrians', 3:'bus', 4:'car', 5:'van', 6:'truck', 7:'motorbike',8:'bicycle'}

ids = [x for x in range(1,41)]  #1-100
#for t in np.arange(seq.radar_init_timestamp, seq.radar_end_timestamp+dt, dt):
for t in ids:
    output = seq.get_radar(t)
    
    if output != {}:
        radar = output['sensors']['radar_cartesian']
        annotations = output['annotations']['radar_cartesian']
        

        radar_id = output['id_radar']
        #'''
        predictions = predictor(radar)
        predictions = predictions["instances"].to("cpu")
        boxes = predictions.pred_boxes
        classes = predictions.pred_classes.numpy()
        #print("predicted:",predictions)
        #continue
        radar_id = output['id_radar']

        objects = []

        for i,box in enumerate(boxes):
            if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RRPN':
                bb, angle = box.numpy()[:4], box.numpy()[4]        
            else:
                bb, angle = box.numpy(), 0   
                bb[2] = bb[2] - bb[0]
                bb[3] = bb[3] - bb[1]
            if not multi_class:
                objects.append({'bbox': {'position': bb, 'rotation': angle}, 'class_name': 'vehicle'})
            else:
                objects.append({'bbox': {'position': bb, 'rotation': angle}, 'class_name':0})#category_to_id[classes[i]] })
        #print("objects",objects)
        #'''

        for obj in annotations:
            obj['class_name'] = 'bus'
            
        radar = seq.vis(radar, objects, color=(255,0,0))
        radar = seq.vis(radar, annotations, color=(0,255,0))
        file_name = saveDir + '/' + str(radar_id)+'.png'
        io.imsave(file_name, radar)

        #cv2.imshow(str(radar_id), radar)
        #cv2.waitKey(0)
    

