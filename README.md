# YOLO-V3 Object detection algorithm in PyTorch

This repository contains code for a object detector based on YOLOv3: An Incremental Improvement, implementedin PyTorch. The code is based on the official code of YOLO v3, as well as a PyTorch port of the original code, by marvis. 


## Requirements 
1. Python 3.5
2. OpenCV
3. PyTorch 0.4

To run the algorithm we will need the official weightsfile for coco which can be downloaded [here](https://pjreddie.com/media/files/yolov3.weights). If you are on linux you can type:

```
wget https://pjreddie.com/media/files/yolov3.weights  
```

### Detection on images

For images here is how the detector works:


![Detection](https://github.com/manasmacherla/yolo_object_detection_pytorch/blob/master/det1/det_dog-cycle-car.png) 

To run the detector on an image or a directory containing images:

```
python detect.py --images imgs --det det1 
```

### Detection on videos

To run the detector on a video:

```
python video.py --video video.avi
```

I have run the algorithm on a traffic video. Here is how it worked. Please go to the link: https://www.youtube.com/watch?v=Q5BRowrCAbg



### References:

For more information about the YOLO-V3 algorithm, please go through this paper: https://pjreddie.com/media/files/papers/YOLOv3.pdf

An excellent blog about YOLO-V3 by Ayoosh Kathuria: https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/
