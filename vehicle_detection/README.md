
This code base supports multiple experiment cases from our paper: 

```
cd src_cs
```

1. Baseline standard CS can be done using:

```
matlab -nodesktop -nosplash -r "run compressive_sensing_radar_radiate_polar()"
```
line 7,8 needs to be modified for different sampling rates and measurement matrices (currently BPD, BPBD and Gauss are supported). 

2. RADAR PointCloud baseline:

```
bash main_pcd_20.sh
```

3. Multirate Faster-RCNN based RADAR compression:

```
bash main_multirate_20.sh
```

4. Multirate Faster-RCNN based RADAR compression with tracking:

```
bash main_multirate_kalman_20.sh
```

5. Multirate YoloV5 based RADAR compression:

```
bash main_yolov5_multirate_20.sh
bash  coco_to_yolo.sh
cd yolov5
python3 val.py --data=data/radcoco.yaml --weights=runs/train/exp7/weights/best.pt --imgsz=1280 --batch-size=
```

6. Multirate YoloV5 based RADAR compression with tracking:

```
bash main_yolov5_multirate_kalman_20.sh
```

7. Ablation with anchor sampling (quantization based anchor)

```
bash main_yolov5_multirate_anchor_20.sh
```

Finally, we also tested with DeforDETR model:

```
bash main_DeforDETR_multirate_20.sh
main_DeforDETR_multirate_kalman_20.sh
```








