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

import argparse

parser = argparse.ArgumentParser(description='Arguments for detectron')
parser.add_argument('--npy_name',type=int, default =1, help='radar id to process')
args = parser.parse_args()


sequence = 'city_3_7'

net_output = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence+'/30-net_output-same-meas'
saveDir = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence+'/30-net_output_idx-same-meas'

if not os.path.isdir(saveDir):
    os.mkdir(saveDir)

npy_name = net_output+ '/' + str(args.npy_name)+'.npy'

net_output = np.load(npy_name)

print(net_output)

obj_rows = []
obj_columns = []
objs = []

dx_9 = [-1,0,1]
dy_9 = [-1,0,1]

dx_25 = [-2,-1,0,1,2]
dy_25 = [-2,-1,0,1,2]


def valid(x,y):
    if x <0 or x>=23 or y<0 or y>=23:
        return False
    else:
        return True

for box in net_output:
    box_x = box[0]
    box_y = box[1]

    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]

    if box[2] >=50 or box[3]>=50:
        total = 25
        dx = dx_25
        dy = dy_25
    else:
        total = 9
        dx = dx_9
        dy = dy_9

    x_idx = int((box_x/1152)*23)
    y_idx = int((box_y/1152)*23)

    for x in dx:
        for y in dy:
            if valid(x_idx+x, y_idx+y):
                objs.append((x_idx+x, y_idx+y))

#'''
objs = list(set(objs))
objs = np.array(objs)
obj_rows = objs[:,1] #y is row
obj_columns = objs[:,0] #x is column

file_name_row = saveDir +'/'+ str(args.npy_name)+'_row.mat'
sio.savemat(file_name_row, {'obj_rows': obj_rows})#, {'pcd_columns': pcd_columns})
file_name_column = saveDir +'/'+ str(args.npy_name)+'_column.mat'
sio.savemat(file_name_column, {'obj_columns': obj_columns})
#'''
