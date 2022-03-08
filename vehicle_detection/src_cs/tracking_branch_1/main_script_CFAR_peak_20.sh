echo "-----------started processing first image-----------"

#radarDir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1/scene3/final-img-rad-info/'
#saveDir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1/scene3/radar-pcd-data/'


scenes='city_3_7 night_1_4 motorway_2_2 snow_1_0 tiny_foggy'


#scene="night_1_4"
folder='20-final-rad-info-polar-cfar-3'
sr=20
for scene in $scenes
do
for i in {2..21}
do
   echo "Processing $i th image"
   ((prev = i - 1))
   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection
   python3 Polar_to_Cartesian_script.py --radar_id=$prev --scene=$scene --folder=$folder 
   if [[ $i -lt 21 ]]
   then
   #python3 detection_script_polar_final.py  --radar_id=$prev --scene=$scene --folder=$folder --sr=$sr
   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection/src_cs
   python3 script_peak_detect_cfar.py --npy_name=$prev --scene=$scene --sr=$sr --folder=$folder
   echo "Processing $i th image $scene"
   matlab -nodesktop -nosplash -c 3221@harrison.ece.utexas.edu -r "var1='$scene';var2='$folder'; run compressed_sensing_radar_pcd_bash_polar_final_1($i,var1,var2,$sr)"
   fi
done
done





echo "----------finished processing all images------------"