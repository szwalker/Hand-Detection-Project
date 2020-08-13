# Real-time Hand Dection Project

## Data Spliting
The data from EgoHannds has been splited into a training set that contains 80 percent of the original dataset and a validation set that contains the remaining 20 percent of the data. The data for training and validation are selected randomly.

## Traning
I trained the model using the pretrained Faster-RCNN with ResNet 50 FPN model and using SGD(Stochastic Gradient Descent) optimizer. During training, I have experimented with 3 differ- ent sets of hyper-parameters:
* learning rate: 0.005, momentum 0.9, weight decay: 0.0005. The learning rate reaches zero after completing the 16th epoch.
* learning rate: 0.001.
* learning rate: 0.007.

Image: Training Loss Over Epoch for the First Model

## Model Evalution
I let the program to randomly picked 5 images from the dataset and then performed the evaluation on it (see `evaluation.py`).

Here are my randomly chosen pictures:

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

Thank you for reading the above information.
