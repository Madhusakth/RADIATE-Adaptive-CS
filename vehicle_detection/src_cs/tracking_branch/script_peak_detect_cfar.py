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
parser.add_argument('--scene',type=str, default = 'tiny_foggy', help='data scene number')
parser.add_argument('--folder',type=str, default = '20-final-rad-info-polar-test-1', help='data scene number')
parser.add_argument('--npy_name',type=int, default =1, help='radar id to process')
parser.add_argument('--sr',type=int, default =10, help='sampling rate')
args = parser.parse_args()

sequence = args.scene

if args.sr==10:
    output_folder='10-net_output_idx-polar'
elif args.sr==20:
    output_folder='20-net_output_idx-polar'
else:
    output_folder='30-net_output_idx-polar'

saveDir = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence+'/'+output_folder #10-net_output_idx-polar'

print(saveDir)

data_dir = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'

orig_dir = data_dir+args.scene+'/Navtech_Polar'
recons_dir = data_dir+args.scene+'/'+args.folder

orig_data_path = os.path.join(orig_dir,'*png')
orig_files = sorted(glob.glob(orig_data_path))

recons_data_path = os.path.join(recons_dir,'*png')
recons_files = sorted(glob.glob(recons_data_path))

print(orig_files[args.npy_name-1])
Xorig = cv2.imread(orig_files[args.npy_name-1], cv2.IMREAD_GRAYSCALE)
print(recons_files[args.npy_name-1])
Xrecons = cv2.imread(recons_files[args.npy_name-1], cv2.IMREAD_GRAYSCALE)

Xorig = Xorig.T
Xrecons = Xrecons.T
Xorig_mask = np.zeros((Xorig.shape[0], Xorig.shape[1]))
for row in range(Xorig.shape[0]):
    peak_idx = detect_peaks(Xorig[row], num_train=20, num_guard=2, rate_fa=0.025) #300, 50, 1e-3 #300, 100, 0.2e-2
    Xorig_mask[row,peak_idx] = 1
Xorig_pcd = Xorig*Xorig_mask

Xrecons_mask = np.zeros((Xrecons.shape[0], Xrecons.shape[1]))
for row in range(Xrecons.shape[0]):
    peak_idx = detect_peaks(Xrecons[row], num_train=20, num_guard=2, rate_fa=0.025) #300, 50, 1e-3 #300, 100, 0.2e-2
    Xrecons_mask[row,peak_idx] = 1
Xrecons_pcd = Xrecons*Xrecons_mask
print("pcd for Xorig  Xrecons:", len(np.where(Xorig_mask)[0]), len(np.where(Xrecons_mask)[0]))


objs = []

idxs = np.where(Xrecons_mask)

for i in range(len(idxs[0])):
    objs.append((int((idxs[0][i]/400) *20),int((idxs[1][i]/576) *12)))

objs = list(set(objs))
print("peak points", len(objs))
objs = np.array(objs)

if len(objs)==0:
    obj_rows = []
    obj_columns = []
else:
    obj_rows = objs[:,0] #y is row
    obj_columns = objs[:,1] #x is column

file_name_row = saveDir +'/'+ str(args.npy_name)+'_row.mat'
sio.savemat(file_name_row, {'obj_rows': obj_rows})#, {'pcd_columns': pcd_columns})
file_name_column = saveDir +'/'+ str(args.npy_name)+'_column.mat'
sio.savemat(file_name_column, {'obj_columns': obj_columns})

'''
for num,images in enumerate(recons_files):
    print(images)
    orig_file = orig_dir + images[-21:-4]+'.png'
    Xorig = cv2.imread(orig_file, cv2.IMREAD_GRAYSCALE)


    Xrecons_mat = sio.loadmat(images)
    Xrecons = np.array(Xrecons_mat['final_A_meta'])
    X_snr = np.array(Xrecons_mat['snrs'])
    print("SNR:", np.mean(X_snr))

    print(Xorig.shape, Xrecons.shape)


    Xorig_meta = Xorig[:,:11]
    Xorig_radar = Xorig[:,11:3711]
    Xorig_mask = np.zeros((Xorig_radar.shape[0], Xorig_radar.shape[1]))
    for row in range(Xorig_radar.shape[0]):
        peak_idx = detect_peaks(Xorig_radar[row], num_train=300, num_guard=50, rate_fa=1e-3) #300, 50, 1e-3 #300, 100, 0.2e-2
        Xorig_mask[row,peak_idx] = 1
    Xorig_pcd = Xorig_radar*Xorig_mask


    
    
    
    Xrecons_meta = Xrecons[:,:11]
    Xrecons_radar = Xrecons[:,11:]
    Xrecons_mask = np.zeros((Xrecons_radar.shape[0], Xrecons_radar.shape[1]))
    for row in range(Xrecons_radar.shape[0]):
        peak_idx = detect_peaks(Xrecons_radar[row], num_train=300, num_guard=50, rate_fa=1e-3) #300, 50, 1e-3 #300, 100, 0.2e-2
        Xrecons_mask[row,peak_idx] = 1
    Xrecons_pcd = Xrecons_radar*Xrecons_mask

    print("pcd for Xorig  Xrecons:", len(np.where(Xorig_mask)[0]), len(np.where(Xrecons_mask)[0]))
'''
