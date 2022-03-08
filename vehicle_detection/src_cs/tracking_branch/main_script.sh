echo "-----------started processing first image-----------"

#radarDir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1/scene3/final-img-rad-info/'
#saveDir='/home/ms75986/Desktop/Qualcomm/Radar-Samp/Adaptive-Radar-Acquisition/data/oxford-v1/scene3/radar-pcd-data/'

for i in {2..20}
do
   echo "Processing $i th image"
   ((prev = i - 1))
   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection
   python3 detection_script.py  --radar_id=$prev
   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection/src_cs
   python3 script_peak_detect.py --npy_name=$prev
   echo "Processing $i th image"
   matlab -nodesktop -nosplash -c 3221@harrison.ece.utexas.edu -r "run compressed_sensing_radar_pcd_bash($i)"
done






echo "----------finished processing all images------------"
