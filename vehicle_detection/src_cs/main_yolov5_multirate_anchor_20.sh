echo "-----------started processing first image-----------"
echo "*****************WARNING: Run independent of main_multirate code***************"
echo "*****************WARNING: Activate yolov5-venv****************"

scenes='city_3_7 night_1_4 motorway_2_2 snow_1_0 fog_6_0'
#scenes='fog_6_0' #fog_6_0'
folder='10-yolov5l-test-15-multi-20-anchor-40-quant-anchor' #-occluded' #-no-object-80'
sr=10
full_sr=40
full_sample="1 21 41 61" #"1 6 11 16 21 26 31 36 41 61"

function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    [[ "$LIST" =~ ($DELIMITER|^)$VALUE($DELIMITER|$) ]]
}


for scene in $scenes
do
	for i in {1..41} #2-21 #1-21  22-41
	do
	   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection
	   if [[ $i -gt 1 ]] #1
	   then
	   echo "Processing $i th image"
	   ((prev = i - 1))
	   #deactivate   ##
	   source ../../../detectron2-venv/bin/activate ##
	   python Polar_to_Cartesian_script.py --radar_id=$prev --scene=$scene --folder=$folder
           fi
	   
	   if [[ $i -lt 41 ]] #21
   	   then 
	   if exists_in_list "$full_sample" " " $i; then
		   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection/src_cs
		   echo "full sample"
		   pwd
		   matlab -nodesktop -nosplash -c 3221@harrison.ece.utexas.edu -r "var1='$scene';var2='$folder'; run compressive_sensing_radar_multirate_anchor($i,var1,var2,$full_sr)"
	   else
		   source ../../../RADIATE_yolov5/yolov5/yolov5-venv/bin/activate ##
		   python test_yolov5.py --weights=../../../RADIATE_yolov5/yolov5/runs/train/exp7/weights/best.pt --imgsz=1280 --source=../data/radiate/fog_6_0/Navtech_Cartesian/000001.png --data=../../../RADIATE_yolov5/yolov5/data/radcoco.yaml --radar_id=$prev --scene=$scene --folder=$folder --sr=$sr #--conf-thres=0.5
		   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection/src_cs
		   python3 script_peak_detect_polar_final_inverted_yolov5.py --npy_name=$prev --scene=$scene --sr=$sr
		   echo "Processing $i th image $scene"
		   matlab -nodesktop -nosplash -c 3221@harrison.ece.utexas.edu -r "var1='$scene';var2='$folder'; run compressed_sensing_radar_pcd_bash_polar_yolov5($i,var1,var2,$sr)"
	   fi
           fi
	done
done





echo "----------finished processing all images------------"
