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

import detectron2
from detectron2.structures import BoxMode
from detectron2 import structures
import torch

import math

import argparse

parser = argparse.ArgumentParser(description='Arguments for detectron')
parser.add_argument('--scene',type=str, default = 'tiny_foggy', help='data scene number')
parser.add_argument('--npy_name',type=int, default =1, help='radar id to process')
parser.add_argument('--sr',type=int, default =10, help='sampling rate')
args = parser.parse_args()


sequence = args.scene #'city_3_7'

if args.sr==10:
    input_folder='10-net_output-polar'
    output_folder='10-net_output_idx-polar'
elif args.sr==20:
    input_folder='20-net_output-polar'
    output_folder='20-net_output_idx-polar'
else:
    input_folder='30-net_output-polar'
    output_folder='30-net_output_idx-polar'



net_output = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence+'/'+input_folder #10-net_output-polar'
saveDir = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence+'/'+output_folder #10-net_output_idx-polar'

print(net_output)
print(saveDir)

if not os.path.isdir(saveDir):
    os.mkdir(saveDir)

npy_name = net_output+ '/' + str(args.npy_name)+'.npy'
boxes = np.load(npy_name)
print(boxes)

track = False
if args.npy_name >= 2:
    prev_npy_name = net_output+ '/' + str(args.npy_name-1)+'.npy'
    prev_boxes = np.load(prev_npy_name)
    print(prev_boxes)
    track = True


obj_rows = []
obj_columns = []
objs = []


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

def euclidean(box, prev_box):
    x_box = box[0] + (box[2]-box[0])/2
    y_box = box[1] + (box[3]-box[1])/2

    x_box_prev = prev_box[0] + (prev_box[2]-prev_box[0])/2
    y_box_prev = prev_box[1] + (prev_box[3]-prev_box[1])/2

    return np.sqrt((x_box-x_box_prev)**2+(y_box-y_box_prev)**2)

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


    boxes = np.array(boxes.tensor)
    prev_boxes = np.array(prev_boxes.tensor)
    euc_box = np.zeros((len(boxes),len(prev_boxes)))
    for i in range(len(boxes)):
        for j in range(len(prev_boxes)):
            euc_box[i][j] = euclidean(boxes[i],prev_boxes[j])

    print(euc_box)
    matched_idxs = linear_assignment(euc_box)
    print(matched_idxs)

    matches_final = []
    for ele in matched_idxs:
        if euc_box[ele[0]][ele[1]] <50:
            matches_final.append([ele[0],ele[1]])
    print("Using sklearn and euc:",matches_final)

    #return final_pair
    return matches_final

def polar_in_radar(box):
    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]

    box_x = box[0]
    box_y = box[1]

    box_x += box[2]/2
    box_y += box[3]/2


    box_x = box_x - 576
    box_y = box_y - 576

    box_x,box_y = rotate(box_x,box_y,np.pi/2)

    r,theta = cart_to_polar(box_x,box_y)


    if theta <0:
        theta +=360

    return r,theta

if track:
    pairs = track_boxes(structures.Boxes(torch.from_numpy(boxes)),structures.Boxes(torch.from_numpy(prev_boxes)))

for num in range(len(boxes)):

    r,theta = polar_in_radar(boxes[num])
    print("r, theta:",r,theta)
    box = boxes[num]

    x_idx = int((r/576)*12)
    y_idx = int((theta/(0.9*400))*20)
    print("x_idx,y_idx:",x_idx,y_idx)

    track_box = False
    same, away,toward, left,right = False, False, False, False, False

    if track:
        for pair in pairs:
            if num == pair[0]:
                track_box=True
                break
            else:
                track_box=False

        if track_box:

            prev_r,prev_theta = polar_in_radar(prev_boxes[pair[1]])
            prev_x_idx = int((prev_r/576)*12)
            prev_y_idx = int((prev_theta/(0.9*400))*20)
            print(prev_x_idx, prev_y_idx)

            away_cases = [(prev_x_idx+1,prev_y_idx-1), (prev_x_idx+1,prev_y_idx), (prev_x_idx+1,prev_y_idx+1)]
            toward_cases = [(prev_x_idx-1,prev_y_idx-1), (prev_x_idx-1,prev_y_idx), (prev_x_idx-1,prev_y_idx+1)]
            left_case = [(prev_x_idx,prev_y_idx+1)]
            right_case = [(prev_x_idx, prev_y_idx-1)]

            if x_idx == prev_x_idx and y_idx == prev_y_idx:
                same = True
            elif (x_idx,y_idx) in away_cases:
                away = True
            elif (x_idx,y_idx) in toward_cases:
                toward = True
            elif (x_idx,y_idx) in left_case:
                left = True
            elif (x_idx,y_idx) in right_case:
                right = True

            #if r > prev_r:
            #    away=True
            #else:
            #    away = False
            #print("r, prev_r:",r,prev_r)


    if x_idx < 6 and box[2] <=50 and box[3] <=50:
        dx = dx_9
        dy = dy_9

        for x in dx:
            for y in dy:
                if valid(x_idx+x, y_idx+y):
                    objs.append(transform(x_idx+x, y_idx+y))

    else:

        if track_box and away:
            dx = [-1,0,1,2]
            print("*********** tracking and away ***********")
        elif track_box and toward:
            dx = [-2,-1,0,1]
            print("*********** tracking and towards ***********")
        elif track_box and (left or right):
            dx = [-1,0,1]
            print("*********** tracking and side ***********")
        #elif track_box and same and box[2] <=75 and box[3] <=75:
        #    dx = [-1,0,1]
        #    print("*********** tracking and same ***********")
        else:
            dx = dx_25
            print("*********** not tracking ***********")

        dy = dy_25

        for x in dx:
            if x <=0:
                dy = dy_9
            else:
                dy = dy_25
            for y in dy:
                if valid(x_idx+x, y_idx+y):
                    objs.append(transform(x_idx+x, y_idx+y))


    if x_idx <= 1:
        print("************* close to object *************")
        dx = dx_angle
        dy = dy_angle
        for x in dx:
            for y in dy:
                if valid(x_idx+x, y_idx+y):
                    objs.append(transform(x_idx+x, y_idx+y))

    print(objs)


#print(objs)

objs = list(set(objs))
objs = np.array(objs)

if len(objs)==0:
    obj_rows = []
    obj_columns = []
else:
    obj_rows = objs[:,1] #y is row
    obj_columns = objs[:,0] #x is column

#obj_rows = objs[:,1] #y is row
#obj_columns = objs[:,0] #x is column

file_name_row = saveDir +'/'+ str(args.npy_name)+'_row.mat'
sio.savemat(file_name_row, {'obj_rows': obj_rows})#, {'pcd_columns': pcd_columns})
file_name_column = saveDir +'/'+ str(args.npy_name)+'_column.mat'
sio.savemat(file_name_column, {'obj_columns': obj_columns})
