from python import radar
import matplotlib.pyplot as plt
import glob
import os
import imageio
import cv2
import numpy as np
import scipy.io as sio
from scripts.cfar import detect_peaks
from skimage import io
from scipy import ndimage
from scipy.signal import find_peaks
import pickle

import math

import sys 
sys.path.append('..')

from kalmanfilter import KalmanFilter
import helpers
import tracker

from detectron2.structures import BoxMode
from detectron2 import structures
import torch
from sklearn.utils.linear_assignment_ import linear_assignment

# radiate sdk
import sys
sys.path.insert(0, '../../')
import radiate


import argparse

parser = argparse.ArgumentParser(description='Arguments for detectron')
parser.add_argument('--scene',type=str, default = 'tiny_foggy', help='data scene number')
parser.add_argument('--npy_name',type=int, default =1, help='radar id to process')
parser.add_argument('--sr',type=int, default =10, help='sampling rate')
parser.add_argument('--folder',type=str, default = 'test-1', help='folder name')
args = parser.parse_args()


sequence = args.scene #'city_3_7'

if args.sr==10:
    input_folder='10-net_output-polar-kalman'
    output_folder='10-net_output_idx-polar-kalman'
elif args.sr==20:
    input_folder='20-net_output-polar-kalman'
    output_folder='20-net_output_idx-polar-kalman'
else:
    input_folder='30-net_output-polar-kalman'
    output_folder='30-net_output_idx-polar-kalman'



net_output = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence+'/'+input_folder #10-net_output-polar'
saveDir = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence+'/'+output_folder #10-net_output_idx-polar'
print(net_output)
print(saveDir)

root_path = '../../data/radiate/'
sequence_name=args.scene
reconst_path_seq = args.folder+'/radar-cart-img'
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/config/config.yaml',reconst_path = reconst_path_seq)

min_tracks = 3 #2  test-4 3
max_age = -3 #-2   test-4 -3

if not os.path.isdir(saveDir):
    os.mkdir(saveDir)

npy_name = net_output+ '/' + str(args.npy_name)+'.npy'

net_output = np.load(npy_name)

#print(net_output)


dx_9 = [-1,0,1]
dy_9 = [-1,0,1]

dx_25 = [-2,-1,0,1,2]
dy_25 = [-2,-1,0,1,2]

dx_angle = [-1,0]
dy_angle = [-3,-2,-1,0,1,2,3]

def valid(x,y):
    if x <0 or x>=12:
        return False
    return True

def transform(x,y):
    if y<0 or y>=20:
        if y <0:
           y +=20
        elif y >=20:
            y -=20
    
    return (x,y)

def rotate(x,y,theta):
    R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    X_Y = np.array([x,y])
    return np.matmul(R,X_Y)

def cart_to_polar(x,y):
    z = x+1j*y
    rho = np.abs(z)
    phi = (np.angle(z)*180)/np.pi
    #rho = np.sqrt(x**2 + y**2)
    #phi = math.degrees(math.atan2(y,x)) #np.arctan2(y, x)
    return(rho, phi)

'''
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
'''

def euclidean(box, prev_box):
    x_box = box[0] + (box[2]-box[0])/2
    y_box = box[1] + (box[3]-box[1])/2

    x_box_prev = prev_box[0] + (prev_box[2]-prev_box[0])/2
    y_box_prev = prev_box[1] + (prev_box[3]-prev_box[1])/2

    return np.sqrt((x_box-x_box_prev)**2+(y_box-y_box_prev)**2)

def track_boxes(boxes,prev_boxes):

    IOUs = structures.pairwise_iou(boxes,prev_boxes)
    #print(IOUs)

    from sklearn.utils.linear_assignment_ import linear_assignment
    matched_idxs = linear_assignment(-IOUs)
    #print(matched_idxs)

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
    #print(matches)

    #print(final_matches)
    
    final_pair = []
    for item in final_matches:
        final_pair.append([final_matches[item][0],item])
    #print("using custom algo:", final_pair)


    boxes = np.array(boxes.tensor)
    prev_boxes = np.array(prev_boxes.tensor)
    euc_box = np.zeros((len(boxes),len(prev_boxes)))
    for i in range(len(boxes)):
        for j in range(len(prev_boxes)):
            euc_box[i][j] = euclidean(boxes[i],prev_boxes[j])

    #print(euc_box)
    matched_idxs = linear_assignment(euc_box)
    #print(matched_idxs)
    '''
    matches_final = []
    for ele in matched_idxs:
        if euc_box[ele[0]][ele[1]] <75:
            matches_final.append([ele[0],ele[1]])
    print("Using sklearn and euc:",matches_final)
    '''

    #return final_pair
    return matches_final


vis = False

full_samp = [1,21,41,61]
def kalman_track():
    final_box = []
    #Generate Kalman Filters
    if args.npy_name in full_samp: #== 1:
        kalman = []
        for idx, box in enumerate(net_output):
                kf = KalmanFilter()
                box_x = box[0] + (box[2]-box[0])/2
                box_y = box[1] + (box[3]-box[1])/2

                tmp_trk = tracker.Tracker() # Create a new tracker
                tmp_trk.R_scaler = 1.0/16
                # Update measurement noise covariance matrix
                tmp_trk.update_R()

                x = np.array([[box[0], 0, box[1], 0, box[2]-box[0], 0, box[3]-box[1], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()   ####
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[0]+xx[4], xx[2]+xx[6]]
                #kalman.append([tmp_trk,np.array(box),0,0])
                kalman.append([tmp_trk,xx,0,0]) ##

                final_box.append((box,0))

        pickle_name = saveDir +'/'+ 'kalman.pkl'
        open_file = open(pickle_name, "wb")
        pickle.dump(kalman, open_file)
        open_file.close()
        print("**********Initialised Kalman filter*********")
        return final_box
        #return net_output

    elif args.npy_name not in full_samp: #> 1:
        #Load the saved kalman filters 
        pickle_name = saveDir +'/'+ 'kalman.pkl'
        open_file = open(pickle_name, "rb")
        kalman = pickle.load(open_file)
        open_file.close()

        boxes = net_output

        print(kalman)

        prev_boxes = []
        for i in range(len(kalman)):
            prev_boxes.append(kalman[i][1])
        print("prev_boxes:", args.npy_name,prev_boxes)

        #If the previous frame removed all tracked items, re-init kalman with new predictions
        if len(prev_boxes) == 0:
            kalman = []
            for idx, box in enumerate(net_output):
                    kf = KalmanFilter()
                    box_x = box[0] + (box[2]-box[0])/2
                    box_y = box[1] + (box[3]-box[1])/2

                    tmp_trk = tracker.Tracker() # Create a new tracker
                    tmp_trk.R_scaler = 1.0/16
                    # Update measurement noise covariance matrix
                    tmp_trk.update_R()

                    x = np.array([[box[0], 0, box[1], 0, box[2]-box[0], 0, box[3]-box[1], 0]]).T
                    tmp_trk.x_state = x
                    tmp_trk.predict_only()   ####
                    xx = tmp_trk.x_state
                    xx = xx.T[0].tolist()
                    xx =[xx[0], xx[2], xx[0]+xx[4], xx[2]+xx[6]]
                    kalman.append([tmp_trk,xx,0,0]) ##

                    final_box.append((box,0))

            pickle_name = saveDir +'/'+ 'kalman.pkl'
            open_file = open(pickle_name, "wb")
            pickle.dump(kalman, open_file)
            open_file.close()
            print("**********Initialised Kalman filter*********")
            return final_box

        prev_boxes = structures.Boxes(prev_boxes)
        boxes = structures.Boxes(boxes)
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

        matched_detect = []
        matched_predict = []

        for pair in final_pair:
                curr_kalman = kalman[pair[1]][0]
                #box_x = np.array(boxes[pair[0]].tensor)[0][0] + (np.array(boxes[pair[0]].tensor)[0][2] - np.array(boxes[pair[0]].tensor)[0][0])/2
                #box_y = np.array(boxes[pair[0]].tensor)[0][1] + (np.array(boxes[pair[0]].tensor)[0][3] - np.array(boxes[pair[0]].tensor)[0][1])/2
                #print("original:", np.array(boxes[pair[0]].tensor)[0][0] , np.array(boxes[pair[0]].tensor)[0][1])

                #box = np.array(boxes[pair[0]].tensor)[0]
                box = np.array(boxes.tensor)[pair[0]]


                matched_detect.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
                predict_box = kalman[pair[1]][1]
                matched_predict.append([predict_box[0], predict_box[1], predict_box[2]-predict_box[0], predict_box[3]-predict_box[1]])

                print("original:", box[0],box[1])
                #tmp_trk = tracker.Tracker() # Create a new tracker
                tmp_trk = curr_kalman
                x = np.array([[box[0], 0, box[1], 0, box[2]-box[0], 0, box[3]-box[1], 0]]).T
                #tmp_trk.x_state = x
                box[2] = box[2] - box[0]
                box[3] = box[3] - box[1]
                box = np.array([box[0], box[1], box[2], box[3]])
                tmp_trk.kalman_filter(box.T)
                #xx = tmp_trk.predict_only()
                #xx = xx.T[0].tolist()
                tmp_trk.predict_only()
                xx = tmp_trk.x_state.T[0].tolist()
                #xx =[xx[0], xx[2], xx[4], xx[6]]
                xx =[xx[0], xx[2], xx[0]+xx[4], xx[2]+xx[6]] ####
                print("predicted:", xx)
                #predicted = curr_kalman.predict(box_x, box_y)
                #print("predicted:", predicted)
                kalman[pair[1]][0] = tmp_trk
                #kalman[pair[1]][1] = np.array(boxes[pair[0]].tensor)[0]
                #kalman[pair[1]][1] = np.array(boxes.tensor)[pair[0]] ####
                kalman[pair[1]][1] = xx
                kalman[pair[1]][2] +=1

                if kalman[pair[1]][2] > min_tracks:
                    #final_box.append([xx[0],xx[1],xx[0]+xx[2],xx[1]+xx[3]])
                    final_box.append(([xx[0],xx[1],xx[2],xx[3]],1))
                    print("*********using predicted position*********")
                else:
                    #final_box.append(np.array(boxes[pair[0]].tensor)[0])
                    final_box.append((np.array(boxes.tensor)[pair[0]],0))
                    print("*********tracked but using original**********")

        for i in unmatched_trackers:
                kalman[i][3] -=1


            #'''
        for i in unmatched_detections:
                box = np.array(boxes[i].tensor)[0]
                matched_detect.append([box[0], box[1], box[2]-box[0], box[3]-box[1]])
                '''
                tmp_trk = tracker.Tracker() # Create a new tracker
                tmp_trk.R_scaler = 1.0/16
                # Update measurement noise covariance matrix
                tmp_trk.update_R()
                x = np.array([[box[0], 0, box[1], 0, box[2]-box[0], 0, box[3]-box[1], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()  ####
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                print("undetected predicted before:", xx)
                #xx =[xx[0], xx[2], xx[0]+xx[4], xx[2]+xx[6]] ################################################[xx[0], xx[2], xx[4], xx[6]]
                xx = [xx[0], xx[2], xx[4], xx[6]]
                #tmp_trk.box = xx
                #kalman.append([tmp_trk,np.array(boxes[i].tensor)[0],0,0])
                #kalman.append([tmp_trk,np.array(boxes.tensor)[i],0,0])
                kalman.append([tmp_trk,xx,0,0]) ####
                '''
                final_box.append((box,0))
                print("********using original, non-tracked*********")

        if vis:
                print("Entering visualisation ***************************************************")
                output = seq.get_radar(args.npy_name)

                if output != {}:
                    radar = output['sensors']['radar_cartesian']
                    radar_id = output['id_radar']
                matched_detect = structures.Boxes(matched_detect)
                matched_predict = structures.Boxes(matched_predict)

                detect_objects = []
                predict_objects = []
                for i,box in enumerate(matched_detect):
                    detect_objects.append({'bbox': {'position': box.numpy(), 'rotation': 0}, 'class_name':'vehicle'})

                for i,box in enumerate(matched_predict):
                    predict_objects.append({'bbox': {'position': box.numpy(), 'rotation': 0}, 'class_name':'bus'})

                radar = seq.vis(radar, detect_objects, color=(255,0,0))
                radar = seq.vis(radar, predict_objects, color=(0,255,0))
                saveDir_img = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence+'/'+args.folder+'/'+'kalman_pred_detect'
                #print(saveDir_img)
                if not os.path.isdir(saveDir_img):
                    os.mkdir(saveDir_img)

                file_name = saveDir_img + '/' + str(radar_id)+'.png'
                io.imsave(file_name, radar)

        print("Kalman:", kalman)
        updated_kalman = []
        for i in range(len(kalman)):
                if kalman[i][3] > max_age:
                    updated_kalman.append(kalman[i])
        kalman = updated_kalman
        print("updated kalman:", updated_kalman)

        pickle_name = saveDir +'/'+ 'kalman.pkl'
        open_file = open(pickle_name, "wb")
        pickle.dump(kalman, open_file)
        open_file.close()

        return final_box

def box_to_points(box_x,box_y):
    box_x = box_x - 576
    box_y = box_y - 576
    #box_y = 1152 - box_y - 576

    box_x,box_y = rotate(box_x,box_y,np.pi/2)

    r,theta = cart_to_polar(box_x,box_y)


    if theta <0:
        theta +=360

    #print("r,theta",r,theta)

    x_idx = int((r/576)*12)
    y_idx = int((theta/(0.9*400))*20)
    #print("x_idx,y_idx",x_idx,y_idx)

    return (x_idx, y_idx)



def main():

    obj_rows = []
    obj_columns = []
    objs = []


    final_boxes = kalman_track()
    print(final_boxes)

    for num,item in enumerate(final_boxes):

        box = item[0]
        track_on = item[1]

        box[2] = box[2] - box[0]
        box[3] = box[3] - box[1]
        points = []
        '''
        x_idx, y_idx = box_to_points(box[0],box[1])
        print("up-left corner:", x_idx, y_idx)
        points.append((x_idx,y_idx))

        x_idx, y_idx = box_to_points(box[0]+box[2],box[1])
        print("up-right corner:", x_idx, y_idx)
        points.append((x_idx,y_idx))

        x_idx, y_idx = box_to_points(box[0],box[1]+box[3])
        print("down-left corner:", x_idx, y_idx)
        points.append((x_idx,y_idx))

        x_idx, y_idx = box_to_points(box[0]+box[2],box[1]+box[3])
        print("down-right corner:", x_idx, y_idx)
        points.append((x_idx,y_idx))
        '''

        box_x = box[0]
        box_y = box[1]

        box_x += box[2]/2
        box_y += box[3]/2

        print("box_x,box_y",box_x,box_y)

        x_idx, y_idx = box_to_points(box_x,box_y)
        print("central points:", x_idx, y_idx)
        points.append((x_idx,y_idx))

        print("unique:", len(list(set(points))))
        unique_points = len(list(set(points)))
        '''
        if unique_points < 5 and track_on == 1:
            #4 corner points around centre to account for rotated boxes
            point_x, point_y = box_to_points(box[0]+box[2]/2,box[1])
            points.append((point_x,point_y))
            points.append((point_x,point_y+1))  #additional area around box

            point_x, point_y = box_to_points(box[0]-box[2]/2,box[1])
            points.append((point_x,point_y))
            points.append((point_x,point_y-1))

            point_x, point_y = box_to_points(box[0],box[1]+box[3]/2)
            points.append((point_x,point_y))
            points.append((point_x+1,point_y))

            point_x, point_y = box_to_points(box[0],box[1]-box[3]/2)
            points.append((point_x,point_y))
            points.append((point_x-1,point_y))
            
            for ele in points:
                objs.append(transform(ele[0], ele[1]))
            print("************* tight boxes based on tracking *************")
        '''
        if (track_on ==1 or x_idx > 6) and box[2] <=50 and box[3] <=50:
            dx = dx_9
            dy = dy_9

            for x in dx:
                for y in dy:
                    if valid(x_idx+x, y_idx+y):
                        objs.append(transform(x_idx+x, y_idx+y))
        #elif track_on == 1:
        #    dx = dx_25
        #    dy = dy_9
        #    for x in dx:
                #if x >=0:
                #    dy = dy_9
                #else:
                #    dy = dy_25
        #        for y in dy:
        #            if valid(x_idx+x, y_idx+y):
        #                objs.append(transform(x_idx+x, y_idx+y))
        else:
            dx = dx_25
            dy = dy_25

            for x in dx:
                if x >=0:
                    dy = dy_9
                else:
                    dy = dy_25
                for y in dy:
                    if valid(x_idx+x, y_idx+y):
                        objs.append(transform(x_idx+x, y_idx+y))


        if x_idx <= 1:
            #print("************* close to object *************")
            dx = dx_angle
            dy = dy_angle
            for x in dx:
                for y in dy:
                    if valid(x_idx+x, y_idx+y):
                        objs.append(transform(x_idx+x, y_idx+y))

        #print(objs)


    #'''
    objs = list(set(objs))
    objs = np.array(objs)

    if len(objs)==0:
        obj_rows = []
        obj_columns = []
    else:
        obj_rows = objs[:,1] #y is row
        obj_columns = objs[:,0] #x is column

    file_name_row = saveDir +'/'+ str(args.npy_name)+'_row.mat'
    sio.savemat(file_name_row, {'obj_rows': obj_rows})#, {'pcd_columns': pcd_columns})
    file_name_column = saveDir +'/'+ str(args.npy_name)+'_column.mat'
    sio.savemat(file_name_column, {'obj_columns': obj_columns})
    #'''

main()
