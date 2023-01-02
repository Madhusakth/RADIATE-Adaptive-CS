from python import radar
import matplotlib.pyplot as plt
import glob
import os
#import imageio
import cv2
import numpy as np
import scipy.io as sio
from skimage import io
import argparse

parser = argparse.ArgumentParser(description='Arguments for detectron2.')
parser.add_argument('--scene',type=str, default = 'tiny_foggy', help='data scene number')
parser.add_argument('--folder',type=str, default ='Navtech_Polar', help='front data for True and rear data for False')
parser.add_argument('--radar_id',type=int, default ='mat', help='input radar file as either .mat or .png')
args = parser.parse_args()


# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .17361
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 1152 #1001 #501  # pixels
interpolate_crossover = True



#home_dir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1' 
#data_dir = home_dir+'/scene'+str(args.scene)+'/'+args.folder
data_dir='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+args.scene+'/'+args.folder
save_dir= data_dir + '/radar-cart-img/'

data_path = os.path.join(data_dir,'*png')


files = glob.glob(data_path)
files = sorted(files)
if not os.path.isdir(save_dir):
    os.mkdir(save_dir)
#print(files)
#for num,images in enumerate(files):
if len(files)!=0:
    images = files[args.radar_id-1]
    print(images)
    
    
    X = Xorig = cv2.imread(images, cv2.IMREAD_GRAYSCALE)

    # PCD generation from raw radar data
    fft_data, radar_resolution = radar.load_radar(X.T)
    azimuths = np.load('azimuth.npy')
    radar_resolution = cart_resolution
    cart_img_pcd = radar.radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                        interpolate_crossover)
    file_name = save_dir + images[-10:-4]+'.png'
    io.imsave(file_name, cart_img_pcd)
    
    #continue
    #cv2.imshow("opencv", cart_img_pcd)
    #io.imsave('original.png', cart_img_pcd)
    #cv2.waitKey(0)
