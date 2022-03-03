import matplotlib.pyplot as plt
import glob
import os
import imageio
import cv2
import numpy as np
import scipy.io as sio
from skimage import io
import argparse


data_path ='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/tiny_foggy/Navtech_Cartesian/reconstruct/' 
save_dir='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/tiny_foggy/Navtech_Cartesian/reconstruct/reshaped/'

data_path = os.path.join(data_path,'*png')
files = glob.glob(data_path)

if not os.path.isdir(save_dir):
    os.mkdir(save_dir)

for num,images in enumerate(files):
    print(images)
    X = Xorig = cv2.imread(images, cv2.IMREAD_GRAYSCALE)
    X_reshaped=np.zeros((1152,1152))
    X_reshaped[0:1150,0:1150]=X
    X_reshaped = np.uint8(X_reshaped)
    file_name = save_dir + images[-10:-4]+'.png'
    io.imsave(file_name, X_reshaped)


