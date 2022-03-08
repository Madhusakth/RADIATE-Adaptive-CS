# End-to-end system for object detection from sub-sampled radar data

In this repo, we release the code for our end-to-end sub-sampled radar data acqusition based on radar-based object detection system. 

In this paper, we use the RADIATE (RAdar Dataset In Adverse weaThEr) dataset released by Heriot-Watt University which includes Radar, Lidar, Stereo Camera and GPS/IMU for our experiments. 

Please refer to RADIATE repo for detailed information on how to download and use the radiate sdk: https://github.com/marcelsheeny/radiate_sdk

You can download the RADIATE data into data/ folder 

To run the baseline algorithm, Rad-Info-1 and Rad-Info-2, cd to vehicle_detection/src_cs 

To run standard CS across the test cases used in the paper, for a given sampling rate, 
modify samp_rate variable and the save_folder and run 
```
matlab -nodesktop -nosplash -r "run compressive_sensing_radar_radiate_polar"
```
For example, set saveDir to 10-reconstruct-standard-cs and samp_rate to 0.10 for 10% uniform sampling rate reconstruction. 

To run the proposed algorithm Rad-Info-2, cd to vehicle_detection/src_cs

For 10% sampling: 
```
bash main_script_polar_all_final_10.sh 
```
Similarly, main_script_polar_all_final_20.sh and main_script_polar_all_final_30.sh for 20% and 30% respectively. 





