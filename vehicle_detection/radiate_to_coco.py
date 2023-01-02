import sys
sys.path.append('../../../Defor-DETR/Deformable-DETR/')

import os
import sys
import pickle
import numpy as np
import argparse
from tqdm import tqdm, trange
from cocoplus.coco import COCO_PLUS
import cv2
import json


def parse_args():
    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Converts the NuScenes dataset to COCO format')

    parser.add_argument('--rad_root', default='/home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/data/radiate/',
                        help='RADIATE dataroot')

    parser.add_argument('--folder', default='Navtech_Polar',
                        help='folders to convert to coco format')

    parser.add_argument('--split', default='test',
                        help='dataset split (test, train)')

    parser.add_argument('--scene', default='fog_6_0',
                        help='scenes to convert to coco format')

    parser.add_argument('--out_dir', default='output_data/test/',
                        help='Output directory for the radcoco dataset')

    parser.add_argument('--use_symlinks', default='False',
                        help='Create symlinks to radiate images rather than copying them')

    parser.add_argument('-l', '--logging_level', default='INFO',
                        help='Logging level')

    parser.add_argument('--total',type=int, default =20, 
                        help='Total images per scene')

    args = parser.parse_args()
    return args

one_class = True
two_class = False
def nuscene_cat_to_coco(nusc_ann_name):

    if one_class:
        COCO_CLASSES = {'vehicle': {'id': 1, 'category': 'vehicle', 'supercategory': 'vehicles'}
                }
    elif two_class:
        COCO_CLASSES = {'person': {'id': 1, 'category': 'person', 'supercategory': 'persons'},
                'vehicle': {'id': 2, 'category': 'vehicle', 'supercategory': 'vehicles'}
                }
    else:
        COCO_CLASSES = {'pedestrian': {'id': 1, 'category': 'pedestrian', 'supercategory': 'person'},
                'bicycle': {'id': 8, 'category': 'bicycle', 'supercategory': 'vehicle'},
                'car': {'id': 3, 'category': 'car', 'supercategory': 'vehicle'},
                'motorbike': {'id': 4, 'category': 'motorbike', 'supercategory': 'vehicle'},
                'bus': {'id': 5, 'category': 'bus', 'supercategory': 'vehicle'},
                'truck': {'id': 6, 'category': 'truck', 'supercategory': 'vehicle'},
                'van': {'id': 7, 'category':'van', 'supercategory':'vehicle'},
                'group_of_pedestrians': {'id': 2, 'category': 'group_of_pedestrians', 'supercategory':'person'}
                }


    ## Convert nuscene categories to COCO cat, cat_ids and supercats
    try:
        coco_equivalent = COCO_CLASSES[nusc_ann_name]
    except KeyError:
        return None, None, None

    coco_cat = coco_equivalent['category']
    coco_id = coco_equivalent['id']
    coco_supercat = coco_equivalent['supercategory']

    return coco_cat, coco_id, coco_supercat

def gen_boundingbox(bbox, angle):
        theta = np.deg2rad(-angle)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        points = np.array([[bbox[0], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1]],
                           [bbox[0] + bbox[2], bbox[1] + bbox[3]],
                           [bbox[0], bbox[1] + bbox[3]]]).T

        cx = bbox[0] + bbox[2] / 2
        cy = bbox[1] + bbox[3] / 2
        T = np.array([[cx], [cy]])

        points = points - T
        points = np.matmul(R, points) + T
        points = points.astype(int)

        min_x = np.min(points[0, :])
        min_y = np.min(points[1, :])
        max_x = np.max(points[0, :])
        max_y = np.max(points[1, :])

        return min_x, min_y, max_x, max_y

#-------------------------------------------------------------------------------
def main():
    args = parse_args()

    ## Categories: [category, supercategory, category_id]
    if one_class:
        categories = [['vehicle',     'vehicles',  1]
                ]
    elif two_class:
        categories = [['person',      'persons' ,  1],
                  ['vehicle',     'vehicles',  2]
                ]
    else:
        categories = [['pedestrian',      'person' ,  1],
                  ['group_of_pedestrians',     'person',  2],
                  ['car',         'vehicle',  3],
                  ['motorbike',  'vehicle',  4],
                  ['bus',         'vehicle',  5],
                  ['truck',       'vehicle',  6],
                  ['van',         'vehicle', 7],
                  ['bicycle',     'vehicle', 8]
                  ]

    ## Short split is used for filenames
    anns_file = os.path.join(args.out_dir, 'annotations', 'instances_' + args.split + '.json')

    # get folders depending on dataset_mode
    folders_train = []
    folders_test = []
    root_dir = args.rad_root

    if args.scene == 'ALL':

        for curr_dir in os.listdir(root_dir):
            with open(os.path.join(root_dir, curr_dir, 'meta.json')) as f:
                meta = json.load(f)
            if meta["set"] == "train_good_weather":
                folders_train.append(curr_dir)
            elif meta["set"] == "train_good_and_bad_weather":
                folders_train.append(curr_dir)
            elif meta["set"] == "test":
                folders_test.append(curr_dir)
        folders_test.remove('tiny_foggy')

        if args.split == 'val2017':#'test':
            folders = folders_test
        else:
            folders = folders_train
            #print(folders_test, folders_train, folders)
            #print(categories)

    elif args.scene== 'all':
        folders = ['city_3_7','fog_6_0','snow_1_0','night_1_4','motorway_2_2']
    else:
        folders = [args.scene]

    print("Final folders:", folders)

    coco_dataset = COCO_PLUS(logging_level="INFO")
    coco_dataset.create_new_dataset(dataset_dir=args.out_dir, split=args.split)

    ## add all category in order to have consistency between dataset splits
    for (coco_cat, coco_supercat, coco_cat_id) in categories:
        coco_dataset.addCategory(coco_cat, coco_supercat, coco_cat_id)

    total_anno = 0
    for folder in folders:
        if args.scene=='ALL' and args.folder=='Navtech_Cartesian':# or args.total>20:
            radar_folder = os.path.join(root_dir, folder, args.folder)
        else:
            radar_folder = os.path.join(root_dir, folder, args.folder, 'radar-cart-img') ##### 

        annotation_path = os.path.join(root_dir,
                                       folder, 'annotations', 'annotations.json')
        with open(annotation_path, 'r') as f_annotation:
            annotation = json.load(f_annotation)

        radar_files = os.listdir(radar_folder)
        radar_files.sort()
        
        if args.scene == 'ALL':
            radar_cart_folder = os.path.join(root_dir, folder, 'Navtech_Cartesian')
            total_cart_imgs = len(os.listdir(radar_cart_folder))
            image_total = len(radar_files) if len(radar_files) < total_cart_imgs else total_cart_imgs
            #image_total = len(radar_files)
        else:
            image_total = args.total

        for frame_number in range(image_total):
            filename = os.path.join(
                radar_folder, radar_files[frame_number])
            #print(filename)

            if (not os.path.isfile(filename)):
                print(filename)
                continue
            image = cv2.imread(filename)
            img_height, img_width = 1152, 1152
            sample_anns = []
            for object in annotation:
                    if (object['bboxes'][frame_number]):
                        class_obj = object['class_name']
                        if one_class:
                            if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):
                                total_anno +=1
                                coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco('vehicle')
                        elif two_class:
                            if (class_obj != 'pedestrian' and class_obj != 'group_of_pedestrians'):
                                coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco('vehicle')
                            else:
                                coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco('person')
                        else:
                            coco_cat, coco_cat_id, coco_supercat = nuscene_cat_to_coco(object['class_name'])
                        cat_id = coco_dataset.addCategory(coco_cat, coco_supercat, coco_cat_id)
                        bbox = object['bboxes'][frame_number]['position']
                        angle = object['bboxes'][frame_number]['rotation']
                        min_x, min_y, max_x, max_y = gen_boundingbox(bbox, angle)

                        bbox = [min_x, min_y, max_x - min_x, max_y - min_y]
                        #bbox = [min_x, min_y, max_x, max_y]  #####################

                        coco_ann = coco_dataset.createAnn(bbox, cat_id)
                        sample_anns.append(coco_ann)
            ## Add sample to the COCO dataset
            coco_img_path = coco_dataset.addSample(img=image,
                                           anns=sample_anns,
                                           pointcloud=None,
                                           img_id=None,
                                           other=None,
                                           img_format='Gray',
                                           write_img=True,
                                           )

            ## Uncomment to visualize
            # coco_dataset.showImgAnn(np.asarray(image), sample_anns, bbox_only=True, BGR=False)
    print("Total annotations:", total_anno)
    coco_dataset.saveAnnsToDisk()

if __name__ == '__main__':
    main()
