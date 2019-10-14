# PaddleDetection applied for specific scenarios

We provide some models implemented by PaddlePaddle to detect objects in specific scenarios, users can download the models and use them in these scenarios.

## Vehicle Detection

One of major applications of vehichle detection is traffic monitoring。In this scenary, vehicles to be detected are mostly captured by the cameras mounted on top of traffic light columns.

### 1. Network

The network for detecting vehicles is Yolo V3, the Backbone of which is Dacknet53.

### 2. Accuracy

The accuracy of the model trained and evaluted on our business data is shown as followed:

AP at IoU=.50:.05:.95 is 0.545.
AP at IoU=.50 is 0.764.

### 3. Inference

Users can employ the model to conduct the inference:

```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python -u tools/infer.py -c contrib/VehicleDetection/vehicle_yolov3_darknet.yml \
                         -o weights= \ 
                         --infer_dir contrib/VehicleDetection/demo \
                         --draw_threshold 0.3 \
                         --output_dir contrib/VehicleDetection/demo/output

```

Some inference results are visualized below:

![](VehicleDetection/demo/output/001.jpeg)

![](VehicleDetection/demo/output/005.png)

## Pedestrian Detection

The main applications of pedetestrian detection include intelligent monitoring. In this scenary, photos of pedetestrians are taken by surveillance cameras in public areas, then pedestrian detection are conducted on these photos.

### 1. Network

The network for detecting vehicles is Yolo V3, the Backbone of which is Dacknet53.

### 2. Accuracy

The accuracy of the model trained and evaluted on our business data is shown as followed:

AP at IoU=.50:.05:.95 is 0.518.
AP at IoU=.50 is 0.792.

### 3. Inference

Users can employ the model to conduct the inference:

```
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=$PYTHONPATH:.
python -u tools/infer.py -c contrib/PedestrianDetection/pedestrian_yolov3_darknet.yml \ 
                         -o weights= \
                         --infer_dir contrib/PedestrianDetection/demo \ 
                         --draw_threshold 0.3 \
                         --output_dir contrib/PedestrianDetection/demo/output
```

Some inference results are visualized below:

![](PedestrianDetection/demo/output/001.png)
![](PedestrianDetection/demo/output/004.png)
