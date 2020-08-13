# Real-time Hand Dection Project

## Dataset Intro
EgoHands dataset contains in total 4800 labeled images from 48 Google Glass videos. The
ground-truth labels consist of polygons with hand segmentations and are provided as Matlab
files.

## Data Spliting and Pre-Prossesing
The data from EgoHannds has been splited into a training set that contains 80 percent of the original dataset and a validation set that contains the remaining 20 percent of the data. The data for training and validation are selected randomly. All images with no hands have been filtered out.

## Traning
I trained the model using the pretrained Faster-RCNN with ResNet 50 FPN model and using SGD(Stochastic Gradient Descent) optimizer. During training, I have experimented with 3 differ- ent sets of hyper-parameters:
* learning rate: 0.005, momentum 0.9, weight decay: 0.0005. The learning rate reaches zero after completing the 16th epoch.
* learning rate: 0.001.
* learning rate: 0.007.

<p align="center">
  <img src="https://github.com/szwalker/Hand-Detection-Project/blob/master/imgs/train_loss_over_epoch.png?raw=true">
  <br>Image: Training Loss Over Epoch for the First Model
</p>


## Model Evalution
I let the program to randomly picked 5 images from the dataset and then performed the evaluation on it (see `evaluation.py`).

Here are my randomly chosen pictures:
![e3](https://github.com/szwalker/Hand-Detection-Project/blob/master/detection_samples/e3.jpg?raw=true)
![e5](https://github.com/szwalker/Hand-Detection-Project/blob/master/detection_samples/e5.jpg?raw=true)
![e1](https://github.com/szwalker/Hand-Detection-Project/blob/master/detection_samples/e1.jpg?raw=true)
![e2](https://github.com/szwalker/Hand-Detection-Project/blob/master/detection_samples/e2.jpg?raw=true)
![e4](https://github.com/szwalker/Hand-Detection-Project/blob/master/detection_samples/e4.jpg?raw=true)


Evalution Statistics
```
IoU metric: bbox
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.804
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.986
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.986
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.804
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.281
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.825
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.825
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.813
```

## Source Code and Model Download
Please view the source code that produces the model at the `src` folder. The model file is not in the repository due to its large file size. However, you can download them via Google Drive at [here](https://drive.google.com/file/d/109nM74b_3J3I_XepRVX4diRKh3LUAH0U/view?usp=sharing).

Thank you for reading the above information.
