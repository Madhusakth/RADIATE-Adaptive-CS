import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader, build_detection_train_loader
import os
import json
from detectron2.structures import BoxMode

# radiate sdk
import sys
sys.path.insert(0, '..')
import radiate


import argparse

parser = argparse.ArgumentParser(description='Arguments for detectron')
parser.add_argument('--scene',type=str, default = 'tiny_foggy', help='data scene number')
parser.add_argument('--folder',type=str, default ='Navtech_Polar', help='front data for True and rear data for False')
parser.add_argument('--radar_id',type=int, default = 0, help='radar id to process')
parser.add_argument('--sr',type=int, default = 10, help='sampling rate')
args = parser.parse_args()


# path to the sequence
root_path = '../data/radiate/'
sequence_name = args.scene #'city_3_7'
radar_path = args.folder+'/radar-cart-img' #'30-final-rad-info-same-meas' #'Navtech_Cartesian'


if args.sr==10:
    output_folder='10-net_output-polar-kalman'
elif args.sr==20:
    output_folder='20-net_output-polar-kalman'
else:
    output_folder='30-net_output-polar-kalman'

net_output = '/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/'+sequence_name+'/'+output_folder #10-net_output-polar'
print(net_output)
if not os.path.isdir(net_output):
    os.mkdir(net_output)


network = 'faster_rcnn_R_50_FPN_3x'
setting = 'good_and_bad_weather_radar'

# time (s) to retrieve next frame
dt = 0.25

# load sequence
seq = radiate.Sequence(os.path.join(root_path, sequence_name), config_file='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/config/config.yaml',reconst_path = radar_path)

cfg = get_cfg()
# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(os.path.join('test','config' , network + '.yaml'))
#cfg.MODEL.DEVICE = 'cpu'  #####
#cfg.MODEL.WEIGHTS = os.path.join('weights',  network +'_' + setting + '.pth')
cfg.MODEL.WEIGHTS = 'train_results_latest/faster_rcnn_R_50_FPN_3x_good_and_bad_weather/model_final.pth'
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # only has one class (vehicle)
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.2
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
predictor = DefaultPredictor(cfg)

#timestamps = np.arange(seq.radar_init_timestamp, seq.radar_end_timestamp+dt, dt)
#t = timestamps[args.radar_id-1]


ids = [x for x in range(1,1000)]
t = ids[args.radar_id-1]

output = seq.get_radar(t)


def main():
    if output != {}:
        radar = output['sensors']['radar_cartesian']
        #annotations = output['annotations']['radar_cartesian']
        predictions = predictor(radar)
        predictions = predictions["instances"].to("cpu")
        boxes = predictions.pred_boxes
        boxes = np.array(boxes.tensor)
        #print(annotations)
        npy_name = net_output + '/' + output['id_radar'] + '.npy'
        print("current radar net output:",npy_name)
        np.save(npy_name, boxes)
        print(boxes)
        #return npy_name
        
main()    
'''
        objects = []

        for box in boxes:
            if cfg.MODEL.PROPOSAL_GENERATOR.NAME == 'RRPN':
                bb, angle = box.numpy()[:4], box.numpy()[4]        
            else:
                bb, angle = box.numpy(), 0   
                bb[2] = bb[2] - bb[0]
                bb[3] = bb[3] - bb[1]
            objects.append({'bbox': {'position': bb, 'rotation': angle}, 'class_name': 'vehicle'})
            
        radar = seq.vis(radar, objects, color=(255,0,0))
        #radar = seq.vis(radar, annotations, color=(255,0,0))

        cv2.imshow('radar', radar)
        cv2.waitKey()
'''

