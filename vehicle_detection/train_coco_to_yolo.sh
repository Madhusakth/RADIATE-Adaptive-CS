#rm -r ~/Desktop/Qualcomm/RADIATE_yolov5/data_radiate
cd ~/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection
python3 radiate_to_coco.py --scene=ALL --folder=Navtech_Polar --total=20 --out_dir=../../../RADIATE_yolov5/data_radiate/ --split=train2017
cd ~/Desktop/Qualcomm/RADIATE_yolov5/yolov5/JSON2YOLO/
python general_json2yolo.py 
cp -r new_dir/labels/train2017 ../../data_radiate/labels
cd ~/Desktop/Qualcomm/RADIATE_yolov5/data_radiate
#mkdir images
mkdir images/train2017
mv train2017/* images/train2017/
cp ../datasets/coco/coco_to_yolo.py .
python coco_to_yolo.py 
#cp ../datasets/coco/train2017.txt .
#cp ../datasets/coco/test-dev2017.txt .


