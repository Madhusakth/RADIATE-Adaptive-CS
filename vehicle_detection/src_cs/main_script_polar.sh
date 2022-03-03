echo "-----------started processing first image-----------"

#radarDir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1/scene3/final-img-rad-info/'
#saveDir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1/scene3/radar-pcd-data/'

scene='snow_1_0'
folder='20-final-rad-info-polar-test'

for i in {2..20}
do
   echo "Processing $i th image"
   ((prev = i - 1))
   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection
   python3 Polar_to_Cartesian_script.py --radar_id=$prev --scene=$scene --folder=$folder
   python3 detection_script_polar.py  --radar_id=$prev --scene=$scene --folder=$folder
   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection/src_cs
   python3 script_peak_detect_polar.py --npy_name=$prev --scene=$scene
   echo "Processing $i th image $scene"
   matlab -nodesktop -nosplash -c 3221@harrison.ece.utexas.edu -r "var='$scene'; run compressed_sensing_radar_pcd_bash_polar($i,var)"
done






echo "----------finished processing all images------------"
