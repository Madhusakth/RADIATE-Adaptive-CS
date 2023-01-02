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

parser = argparse.ArgumentParser(description='Arguments for peak detection')
parser.add_argument('--scene',type=str, default = 'tiny_foggy', help='data scene number')
parser.add_argument('--npy_name',type=int, default =1, help='radar id to process')
parser.add_argument('--sr',type=int, default =10, help='sampling rate')
parser.add_argument('--folder',type=str, default = 'test-1', help='folder name')
args = parser.parse_args()


if args.sr==10:
    input_folder='10-net_output-polar'
    output_folder='10-net_output_idx-polar'
elif args.sr==20:
    input_folder='20-net_output-polar'
    output_folder='20-net_output_idx-polar'
else:
    input_folder='30-net_output-polar'
    output_folder='30-net_output_idx-polar'

saveDir = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+args.scene+'/'+output_folder


# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .25
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 501  # pixels
interpolate_crossover = True



data_dir = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+args.scene+'/'+args.folder
data_path = os.path.join(data_dir,'*png')
files = sorted(glob.glob(data_path))

if not os.path.isdir(output_folder):
    os.mkdir(output_folder)


images = [files[args.npy_name-1]]

#for num,images in enumerate(files):
for images in images:
    print(images)
    X = cv2.imread(images, cv2.IMREAD_GRAYSCALE)
    print(X.shape)
    X = X.T
    X_pcd = X
    
    mask = np.zeros((X.shape[0], X.shape[1]))
    for row in range(X.shape[0]):
        peak_idx = detect_peaks(X[row], num_train=50, num_guard=25, rate_fa=1e-2) #300, 50, 1e-3 #300, 100, 0.2e-2
        #peak_idx, _ = find_peaks(cart_img_pcd[row].reshape(501), distance = 75,width = 10)
        mask[row,peak_idx] = 1
        #print("peak_idx =", peak_idx)
    pcd_rows = []
    pcd_columns = []

    X_pcd = X_pcd*mask
    total = 0
    for row in range(20):
        for column in range(12):
            current = X_pcd[row*20:(row+1)*20, column*48: (column+1)*48]
            if current[current !=0].shape[0] != 0:
                #print(current[current !=0].shape)
                #print(row*50, (row+1)*50, column*100, (column+1)*100)
                pcd_rows.append(row)
                pcd_columns.append(column)
                total = total + 1
    print(total)
    print(pcd_rows, pcd_columns)

    file_name_row = saveDir +'/'+ str(args.npy_name)+'_row.mat'
    sio.savemat(file_name_row, {'obj_rows': pcd_rows})#, {'pcd_columns': pcd_columns})
    file_name_column = saveDir +'/'+ str(args.npy_name)+'_column.mat'
    sio.savemat(file_name_column, {'obj_columns': pcd_columns})

