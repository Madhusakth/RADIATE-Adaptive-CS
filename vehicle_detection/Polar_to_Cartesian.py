from python import radar
import matplotlib.pyplot as plt
import glob
import os
import imageio
import cv2
import numpy as np
import scipy.io as sio
from skimage import io
import argparse

parser = argparse.ArgumentParser(description='Arguments for detectron2.')
parser.add_argument('--scene',type=int, default = 1, help='data scene number')
parser.add_argument('--folder',type=str, default ='radar', help='front data for True and rear data for False')
parser.add_argument('--input_type',type=str, default ='mat', help='input radar file as either .mat or .png')
args = parser.parse_args()


# Cartesian Visualsation Setup
# Resolution of the cartesian form of the radar scan in metres per pixel
cart_resolution = .17361
# Cartesian visualisation size (used for both height and width)
cart_pixel_width = 1152 #1001 #501  # pixels
interpolate_crossover = True



#home_dir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1' 
#data_dir = home_dir+'/scene'+str(args.scene)+'/'+args.folder

#test_folders = ['snow_1_0', 'city_3_7']
#test_folders = ['night_1_4', 'junction_3_2', 'snow_1_0', 'junction_1_12', 'rain_3_0', 'fog_6_0', 'city_3_7', 'city_7_0', 'junction_1_10', 'night_1_2', 'night_1_5', 'rain_4_0', 'night_1_0', 'motorway_2_2', 'tiny_foggy', 'junction_2_6', 'junction_1_11', 'fog_8_1']

#train_folders = ['motorway_2_1', 'junction_2_1', 'city_3_3', 'junction_1_4', 'city_3_1', 'junction_1_13', 'junction_1_6', 'junction_1_7', 'city_6_0', 'fog_8_2', 'junction_1_5', 'rural_1_3', 'junction_1_14', 'junction_2_0', 'city_4_0', 'rain_4_1', 'city_1_3', 'junction_3_0', 'junction_2_5', 'city_1_1', 'junction_2_3', 'city_2_0', 'junction_1_8', 'city_1_0', 'night_1_3', 'junction_3_1', 'junction_1_15', 'night_1_1', 'junction_2_2', 'motorway_1_0', 'city_3_2', 'junction_1_2', 'junction_1_0', 'junction_3_3', 'junction_1_9', 'city_5_0', 'fog_8_0', 'city_3_0', 'motorway_2_0', 'junction_1_3', 'rural_1_1', 'rain_2_0', 'junction_1_1'] 
train_folders = ['city_3_7','snow_1_0','night_1_4','fog_6_0','motorway_2_2']

total_snr = 0
for test_fol in train_folders:
    print(test_fol)

    data_dir='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/' + test_fol + '/standard-cs-BPD-40-img-20-1' #standard-cs-Gauss-40-img-10'
    save_dir= data_dir + '/radar-cart-img/'

    if args.input_type =='mat':
        data_path = os.path.join(data_dir,'*mat')
    else:
        data_path = os.path.join(data_dir,'*png')


    files = glob.glob(data_path)

    files = sorted(files)
    #files = files[:40] ###

    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)


    for num,images in enumerate(files):
        #print(images)
        if args.input_type=='mat': 
            Xorig = sio.loadmat(images)
            X = np.array(Xorig['final_A_meta'])
            X_snr = np.array(Xorig['snrs'])
            print("SNR:", np.mean(X_snr))
            total_snr += np.mean(X_snr)
            #continue
        else:
            X = Xorig = cv2.imread(images, cv2.IMREAD_GRAYSCALE)

        # PCD generation from raw radar data
        fft_data, radar_resolution = radar.load_radar(X.T)
        #print(fft_data.shape)
        #fft_data[200:,:] = 255
        azimuths = np.load('azimuth.npy')
        radar_resolution = cart_resolution
        cart_img_pcd = radar.radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width,
                                            interpolate_crossover)
        file_name = save_dir + images[-10:-4]+'.png'
        io.imsave(file_name, cart_img_pcd)
        
        continue
        cv2.imshow("opencv", cart_img_pcd)
        io.imsave('original.png', cart_img_pcd)

    if args.input_type=='mat':
        print(total_snr/(40*5))
    cv2.waitKey(0)
