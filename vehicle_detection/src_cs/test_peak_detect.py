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
# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .25
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 501  # pixels
interpolate_crossover = True

parser = argparse.ArgumentParser(description='Arguments for detectron2.')
parser.add_argument('--scene',type=int, default = 1, help='data scene number')
parser.add_argument('--folder',type=str, default ='radar', help='front data for True and rear data for False')
args = parser.parse_args()

data_dir = '/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/'

orig_dir = data_dir+'scene'+str(args.scene)+'/radar'
recons_dir = data_dir+'scene'+str(args.scene)+'/'+args.folder

recons_data_path = os.path.join(recons_dir,'*mat')
recons_files = glob.glob(recons_data_path)

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

