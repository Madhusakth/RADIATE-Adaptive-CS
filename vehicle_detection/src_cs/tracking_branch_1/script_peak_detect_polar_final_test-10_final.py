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

net_output = np.load(npy_name)

print(net_output)

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


for box in net_output:

    box[2] = box[2] - box[0]
    box[3] = box[3] - box[1]

    box_x = box[0]
    box_y = box[1]

    #print(box_x,box_y)

    box_x += box[2]/2
    box_y += box[3]/2

    #print(rotate(1,1,np.pi/2))

    print(box_x,box_y)

    box_x = box_x - 576
    box_y = box_y - 576
    #box_y = 1152 - box_y - 576

    #print(box_x,box_y)
    #print(rotate(box_x,box_y,np.pi/2))
    box_x,box_y = rotate(box_x,box_y,np.pi/2)

    r,theta = cart_to_polar(box_x,box_y)
    

    if theta <0:
        theta +=360

    print(r,theta)

    #theta = 360 - theta
    #print(r,theta)

    #theta = (theta + 270)%360
    #if theta > 360:
    #    theta 360

    #print(r,theta)
    x_idx = int((r/576)*12)
    y_idx = int((theta/(0.9*400))*20)
    print(x_idx,y_idx)

    if x_idx < 6 and box[2] <=50 and box[3] <=50:
        dx = dx_9
        dy = dy_9

        for x in dx:
            for y in dy:
                if valid(x_idx+x, y_idx+y):
                    objs.append(transform(x_idx+x, y_idx+y))

    else:
        dx = dx_25
        dy = dy_25

        for x in dx:
            if x <=0:
                dy = dy_9
            else:
                dy = dy_25
            for y in dy:
                if valid(x_idx+x, y_idx+y):
                    #print(x_idx+x, y_idx+y)
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


print(objs)

'''
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

'''
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
