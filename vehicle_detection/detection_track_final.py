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


from kalmanfilter import KalmanFilter
from detectron2.structures import BoxMode
from detectron2 import structures
import torch

import numpy as np
import matplotlib.pyplot as plt
import glob
#from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment

import helpers
import tracker


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
sequence_name = 'tiny_foggy' #'snow_1_0' #'tiny_foggy' night_1_4 motorway_2_2
radar_path = 'Navtech_Polar/radar-cart-img' #'Navtech_Cartesian_20' #'final-rad-info' #'reconstruct/reshaped' #'Navtech_Cartesian'

network = 'faster_rcnn_R_50_FPN_3x'
setting = 'good_and_bad_weather_radar'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/config/config.yaml',reconst_path = radar_path)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
cfg.MODEL.DEVICE = 'cpu'
cfg.MODEL.WEIGHTS = os.path.join('weights',  network +'_' + setting + '.pth')
#cfg.MODEL.WEIGHTS = 'train_results/faster_rcnn_R_50_FPN_3x_good_and_bad_weather/model_final.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

saveDir = root_path+ sequence_name + '/' + radar_path + '_annotated_nw_orig'
if not os.path.isdir(saveDir):
    os.mkdir(saveDir)


def track_boxes(boxes,prev_boxes):

    IOUs = structures.pairwise_iou(boxes,prev_boxes)
    print(IOUs)

    from sklearn.utils.linear_assignment_ import linear_assignment
    matched_idxs = linear_assignment(-IOUs)
    print(matched_idxs)

    matches_final = []
    for ele in matched_idxs:
        if IOUs[ele[0]][ele[1]] >0:
            matches_final.append([ele[0],ele[1]])
    print("Using sklearn:",matches_final)

    matches = []
    for row in range(IOUs.shape[0]):
        arg_max = int(np.argmax(IOUs[row,:]).numpy())
        val_row = float(IOUs[row,arg_max].numpy())
        if val_row > 0:
            matches.append([row,arg_max,val_row])

    final_matches = {}
    for item in matches:
        str_item = item[1]
        if str_item not in final_matches:
            final_matches[str_item] = [item[0],item[2]]

        else:
            if item[2] > final_matches[str_item][1]:
                final_matches[str_item] = [item[0],item[2]]
    print(matches)

    print(final_matches)

    final_pair = []
    for item in final_matches:
        final_pair.append([final_matches[item][0],item])
    print("using custom algo:", final_pair)

    return final_pair

def main():

    kalman = []
    updated_kalman = []
    ids = [x for x in range(1,21)]  #1-100
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
        print(boxes)

        if t == 1:
            for idx, box in enumerate(boxes):
                kf = KalmanFilter()
                box_x = box[0] + (box[2]-box[0])/2
                #box[1] = 1152-box[1]
                #box[3] = 1152- box[3]
                box_y = box[1] + (box[3]-box[1])/2

                #kf.predict(box_x, box_y)
                #kalman.append([kf,np.array(box)])

                tmp_trk = tracker.Tracker() # Create a new tracker
                tmp_trk.R_scaler = 1.0/16
                # Update measurement noise covariance matrix
                tmp_trk.update_R()
                
                x = np.array([[box[0], 0, box[1], 0, box[2]-box[0], 0, box[3]-box[1], 0]]).T
                tmp_trk.x_state = x
                #tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                #tmp_trk.box = xx
                kalman.append([tmp_trk,np.array(box),0,0])


        if t > 1:
            prev_boxes = []
            for i in range(len(kalman)):
                prev_boxes.append(kalman[i][1])
            print("prev_boxes:", t,prev_boxes)
            prev_boxes = structures.Boxes(prev_boxes)
            final_pair = track_boxes(boxes,prev_boxes)

            #boxes = np.array(boxes.tensor)
            unmatched_detections = []
            box_pair = []
            for pair in final_pair:
                box_pair.append(pair[0])
            for i in range(len(boxes)):
                if i not in box_pair:
                    unmatched_detections.append(i)

            unmatched_trackers = []
            box_pair = []
            for pair in final_pair:
                box_pair.append(pair[1])
            for i in range(len(prev_boxes)):
                if i not in box_pair:
                    unmatched_trackers.append(i)

            print("unmatched_trackers:", unmatched_trackers)
            print("unmatched_detections:", unmatched_detections)

            for pair in final_pair:
                curr_kalman = kalman[pair[1]][0]
                box_x = np.array(boxes[pair[0]].tensor)[0][0] + (np.array(boxes[pair[0]].tensor)[0][2] - np.array(boxes[pair[0]].tensor)[0][0])/2
                box_y = np.array(boxes[pair[0]].tensor)[0][1] + (np.array(boxes[pair[0]].tensor)[0][3] - np.array(boxes[pair[0]].tensor)[0][1])/2
                print("original:", np.array(boxes[pair[0]].tensor)[0][0] , np.array(boxes[pair[0]].tensor)[0][1])
                
                box = np.array(boxes[pair[0]].tensor)[0]
                #tmp_trk = tracker.Tracker() # Create a new tracker
                tmp_trk = curr_kalman
                x = np.array([[box[0], 0, box[1], 0, box[2]-box[0], 0, box[3]-box[1], 0]]).T
                #tmp_trk.x_state = x
                box[2] = box[2] - box[0] 
                box[3] = box[3] - box[1]
                box = np.array([box[0], box[1], box[2], box[3]])
                tmp_trk.kalman_filter(box.T)
                xx = tmp_trk.predict_only()
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                print("predicted:", xx)
                #predicted = curr_kalman.predict(box_x, box_y)
                #print("predicted:", predicted)
                kalman[pair[1]][0] = tmp_trk
                kalman[pair[1]][1] = np.array(boxes[pair[0]].tensor)[0]
                kalman[pair[1]][2] +=1

            for i in unmatched_trackers:
                kalman[i][3] -=1


            #'''
            for i in unmatched_detections:
                box = np.array(boxes[i].tensor)[0]
                tmp_trk = tracker.Tracker() # Create a new tracker
                tmp_trk.R_scaler = 1.0/16
                # Update measurement noise covariance matrix
                tmp_trk.update_R()
                x = np.array([[box[0], 0, box[1], 0, box[2]-box[0], 0, box[3]-box[1], 0]]).T
                tmp_trk.x_state = x
                #tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                #tmp_trk.box = xx
                kalman.append([tmp_trk,np.array(boxes[i].tensor)[0],0,0])

            print("Kalman:", kalman)
            updated_kalman = []
            for i in range(len(kalman)):
                if kalman[i][3] > -1:
                    updated_kalman.append(kalman[i])
            kalman = updated_kalman
            print("updated kalman:", updated_kalman)

            
        '''
        #cv2.imshow(str(radar_id), radar)
        #cv2.waitKey(0)
        '''
    
main()
