echo "-----------started processing first image-----------"


scenes='city_3_7 night_1_4 motorway_2_2 snow_1_0 fog_6_0'
#scenes='rain_3_0 night_1_5 rain_4_0'
folder='20-DeforDETR-kalman-10-multi-20-anchor-80-03thres'
sr=20
full_sr=80
full_sample="1 21 41 61"

function exists_in_list() {
    LIST=$1
    DELIMITER=$2
    VALUE=$3
    [[ "$LIST" =~ ($DELIMITER|^)$VALUE($DELIMITER|$) ]]
}


for scene in $scenes
do
	for i in {1..21} #1-21  #2-21  22-41
	do
	   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection
           if [[ $i -gt 1 ]]
           then
           echo "Processing $i th image"
           ((prev = i - 1))
           python3 Polar_to_Cartesian_script.py --radar_id=$prev --scene=$scene --folder=$folder
           fi

	   if [[ $i -lt 21 ]]
   	   then 
	   if exists_in_list "$full_sample" " " $i; then
		   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection/src_cs
                   echo "full sample"
                   pwd
		   matlab -nodesktop -nosplash -c 3221@harrison.ece.utexas.edu -r "var1='$scene';var2='$folder'; run compressive_sensing_radar_multirate($i,var1,var2,$full_sr)"
	   else
		   python3 DeforDETR_main.py --with_box_refine --two_stage --backbone=resnet101 --coco_path=../../../Defor-DETR/Deformable-DETR/data/coco/ --eval --resume=../../../Defor-DETR/Deformable-DETR/exp_one_class_plus_plus_model_resnet101_augmented_size_800_hori/checkpoint.pth --batch_size=1 --radar_id=$prev --scene=$scene --folder=$folder --sr=$sr
		   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection/src_cs
		   python3 script_peak_detect_polar_kalman_test.py --npy_name=$prev --scene=$scene --sr=$sr
		   echo "Processing $i th image $scene"
		   matlab -nodesktop -nosplash -c 3221@harrison.ece.utexas.edu -r "var1='$scene';var2='$folder'; run compressed_sensing_radar_pcd_bash_polar_kalman($i,var1,var2,$sr)"
	   fi
           fi
	done
done





echo "----------finished processing all images------------"
