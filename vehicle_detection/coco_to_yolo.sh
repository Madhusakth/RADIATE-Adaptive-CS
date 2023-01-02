rm -r ~/Desktop/Qualcomm/RADIATE_yolov5/data_radiate
cd ~/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection
python3 radiate_to_coco.py --scene=all --folder=standard-cs-40-img-20 --total=40 --out_dir=../../../RADIATE_yolov5/data_radiate/coco/ --split=val2017
cd ~/Desktop/Qualcomm/RADIATE_yolov5/yolov5/JSON2YOLO/
python general_json2yolo.py 
cp -r new_dir/labels/ ../../data_radiate/coco/
cd ~/Desktop/Qualcomm/RADIATE_yolov5/data_radiate/coco
mkdir images
mkdir images/val2017
mv val2017/* images/val2017/
cp ../../datasets/coco/coco_to_yolo.py .
python coco_to_yolo.py 
cp ../../datasets/coco/train2017.txt .
cp ../../datasets/coco/test-dev2017.txt .


