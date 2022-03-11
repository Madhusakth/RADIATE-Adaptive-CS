echo "-----------started processing first image-----------"

scene='city_1_3'
folder='20-train_polar-reconst-test-10'
sr=20

for i in {2..100}
do
   echo "Processing $i th image"
   ((prev = i - 1))
   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection
   python3 Polar_to_Cartesian_script.py --radar_id=$prev --scene=$scene --folder=Navtech_Polar
   python3 create_data_detection_script.py --radar_id=$prev
   cd /home/ms75986/Desktop/Qualcomm/RADIATE/radiate_sdk/vehicle_detection/src_cs
   python3 create_script_peak_detect.py --npy_name=$prev
   echo "Processing $i th image"
   matlab -nodesktop -nosplash -c 3221@harrison.ece.utexas.edu -r "var1='$scene';var2='$folder'; run compressed_sensing_radar_train_bash($i,var1,var2,$sr)"
done






echo "----------finished processing all images------------"
