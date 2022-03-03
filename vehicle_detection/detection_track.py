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

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 4  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'])

debug = True

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


#'''
def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    #From current list of trackers and new detections, output matched detections,
    #unmatchted trackers, unmatched detections.
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
            #det[1] = 1152 - det[1]
            #det[3] = 1152 - det[3]
            #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    


def pipeline(img, z_box, z_box_prev):
    #Pipeline function for detection and tracking
    global frame_count
    global tracker_list
    global max_age
    global min_hits
    global track_id_list
    global debug
    
    frame_count+=1
    
    img_dim = (img.shape[1], img.shape[0])
    
    #z_box = det.get_localization(img) # measurement
    if debug:
       print('Frame:', frame_count)
       
    x_box =[]
    '''
    if debug: 
        for i in range(len(z_box)):
           tmp = z_box[i][0] 
           z_box[i][0] = z_box[i][1]
           z_box[i][1] = tmp 
           tmp = z_box[i][2]
           z_box[i][2] = z_box[i][3]
           z_box[i][3] = tmp
           img1= helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
           plt.imshow(img1)
        plt.show()
    '''
    #if len(tracker_list) > 0:
    #    for trk in tracker_list:
    #        x_box.append(trk.box)
    
    
    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(z_box, z_box_prev, iou_thrd = 0.3)  
    if debug:
         print('Detection: ', z_box)
         print('x_box: ', z_box_prev)
         print('matched:', matched)
         print('unmatched_det:', unmatched_dets)
         print('unmatched_trks:', unmatched_trks)
    
         
    # Deal with matched detections     
    if matched.size >0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            '''
            tmp = z_box[det_idx][0]
            z_box[det_idx][0] = z_box[det_idx][1]
            z_box[det_idx][1] = tmp
            tmp = z_box[det_idx][2]
            z_box[det_idx][2] = z_box[det_idx][3]
            z_box[det_idx][3] = tmp
            '''

            #z[1] = 1152 - z[1]
            #z[2] = z[2] - z[0]
            #z[3] = z[3] = z[1]

            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
            tmp_trk.no_losses = 0
    
    # Deal with unmatched detections      
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            #z[1] = 1152 - z[1]
            #z[2] = z[2] - z[0]
            #z[3] = z[3] = z[1]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            tracker_list.append(tmp_trk)
            x_box.append(xx)
    
    # Deal with unmatched tracks       
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
                   
       
    # The list of tracks to be annotated  
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box
             if debug:
                 print('updated box: ', x_cv2)
                 print()
             img= helpers.draw_box_label(img, x_cv2) # Draw the bounding boxes on the 
                                             # images
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  
    
    for trk in deleted_tracks:
            track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    
    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))
    
       
    return img
#'''

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
        #if t > 1:
        #    pipeline(radar, np.array(boxes.tensor), prev_box)
        #prev_box = np.array(boxes.tensor)
        #continue

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
