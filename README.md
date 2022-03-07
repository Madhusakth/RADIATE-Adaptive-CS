# End-to-end system for object detection from sub-sampled radar data

In this repo, we release the code for our end-to-end sub-sampled radar data acqusition based on radar-based object detection system. 

In this paper, we use the RADIATE (RAdar Dataset In Adverse weaThEr) dataset released by Heriot-Watt University which includes Radar, Lidar, Stereo Camera and GPS/IMU for our experiments. 

Please refer to RADIATE repo for detailed information on how to download and use the radiate sdk: https://github.com/marcelsheeny/radiate_sdk

Once you download the radiate data into the /data folder, please run main_script.sh in vehicle_detection/src_cs folder.  


- [RADIATE Dataset](#radiate-dataset)
  - [Dataset size](#dataset-size)
  - [Comparison with other datasets](#comparison-with-other-datasets)
  - [Sensors](#sensors)
  - [Folder Structure and File Format](#folder-structure-and-file-format)
  - [Sensor Calibration](#sensor-calibration)
  - [Annotation Structure](#annotation-structure)
- [RADIATE SDK](#radiate-sdk)
  - [Installation](#installation)
    - [Dependencies](#dependencies)
  - [How to use](#how-to-use)
    - [Example:](#example)
  - [Vehicle Detection](#vehicle-detection)

