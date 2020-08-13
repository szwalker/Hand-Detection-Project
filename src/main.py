# -*- coding: utf-8 -*-
"""
Author: James (Jiaqi) Liu
"""
'''
from google.colab import drive
drive.mount('/content/drive',force_remount=True)
import os
os.chdir("/content/drive/My Drive/hw6")

!pip install cython
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
!pip install torch
!pip install torchvision
!pip install Pillow
!pip install numpy
!pip install scipy
!pip install matplotlib
'''
import os
import numpy as np
from glob import glob
from PIL import Image,ImageDraw
from scipy.io import loadmat
from matplotlib.path import Path
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
import utils
from engine import train_one_epoch, evaluate

class EgoHands(data.Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        folders = sorted(glob(os.path.join(self.path, "*")))
        self.imgs = []
        self.polygons = []
        for folder in folders:
            # extract image from current folder
            folder_imgs = sorted(glob(os.path.join(folder, "*.jpg")))

            polygon_path = glob(os.path.join(folder, "*.mat"))[0]
            polygon = loadmat(polygon_path)['polygons'][0]
            for i in range(len(polygon)):
                have_hand = False
                for j in range(len(polygon[i])): have_hand = have_hand or polygon[i][j].shape not in [(1, 0), (0, 0)]
                # if the current image contains hand
                if have_hand:
                  # add polygon and add image
                  self.polygons.append(polygon[i])
                  self.imgs.append(folder_imgs[i])
        self.transform = transform

    def __getitem__(self, index):
        # Load image
        img = np.array(Image.open(self.imgs[index]))

        # Compute mask
        polygons = self.polygons[index]
        gt_mask = []
        x, y = np.meshgrid(
            np.arange(img.shape[1]), np.arange(img.shape[0]))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T
        for i, polygon in enumerate(polygons):
            if polygon.size == 0:
                continue
            path = Path(polygon)
            grid = path.contains_points(points)
            grid = grid.reshape((*img.shape[:2]))
            gt_mask.append(np.expand_dims(grid, axis=-1))
        gt_mask = np.concatenate(gt_mask, axis=-1)

        # compute minimal bounding boxes
        boxes = []
        pos = np.where(gt_mask, 1, 0).nonzero()
        for channel in range(len(gt_mask[0][0])):
            mask = [[], []]
            for i in range(len(pos[0])):
                if pos[2][i] == channel:
                    mask[0].append(pos[0][i])
                    mask[1].append(pos[1][i])
            if len(mask[0]) > 0:
                x_min,y_min,x_max,y_max = np.min(mask[1]),np.min(mask[0]),np.max(mask[1]),np.max(mask[0])
                boxes.append([x_min, y_min, x_max, y_max])
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        image_id = torch.tensor([index])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target['iscrowd'] = iscrowd
        if self.transform: img, target = self.transform(img,target)
        return img, target

    def __len__(self):
        return len(self.imgs)

    def get_img(self,index): return self.imgs[index]

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train: transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    # hyper param
    BATCH_SIZE = 8
    EPOCHS = 16

    # environment settings
    torch.manual_seed(0) # not allowing the data to be reproduced
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # load model
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # define optimizer, learning rate scheduler
    optimizer = optim.SGD(net.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
    loss_lst = [None for _ in range(EPOCHS)]
    indices = None

    last_complete_epoch = None
    load_checkpoint = None
    if load_checkpoint is not None:
        # load net state_dict
        net_checkpoint = torch.load("./checkpoint/net_{}.pth".format(load_checkpoint))

        # reload net
        net.load_state_dict(net_checkpoint)

        # reload training and testing distribution
        indices = np.load('./checkpoint/indices.npy',allow_pickle=True)

        # reload loss list
        loss_lst = np.load("./checkpoint/loss_{}.npy".format(load_checkpoint),allow_pickle=True)

        print("reloading complete")

    # load dataset [no validation to save RAM]
    DS_train = EgoHands('./egohands_data/_LABELLED_SAMPLES', get_transform(train=True))
    # DS_val = EgoHands('./egohands_data/_LABELLED_SAMPLES',get_transform(train=False))

    # split data into training set and validation set (80% training, 20% validation)
    cutoff_ind = int(len(DS_train) * 0.8)

    if indices is None:
        indices = torch.randperm(len(DS_train)).tolist()
        np.save("./checkpoint/indices.npy",indices)

    # split dataset into training set and validation set
    DS_train = data.Subset(DS_train, indices[:cutoff_ind])
    # DS_val = data.Subset(DS_val, indices[cutoff_ind:])

    # initilize data loader
    DL_train = data.DataLoader(DS_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=utils.collate_fn,num_workers=3)
    # DL_val = data.DataLoader(DS_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=utils.collate_fn, num_workers=12)

    # parallel computing setup
    GPU_count = torch.cuda.device_count()
    print("currently using: {} GPUs".format(GPU_count))
    if GPU_count > 1:
        # increase parallelism
        net = nn.DataParallel(net)
        print("increased parallelism")
    net.to(device)

    for epoch in range(EPOCHS):
        if load_checkpoint is not None and epoch <= load_checkpoint: continue
        # train for one epoch, printing every 10 iterations
        metric_logger, loss = train_one_epoch(net, optimizer, DL_train, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()

        # record loss
        loss_lst[epoch] = loss/len(DL_train)

        # store checkpoints
        torch.save(net.state_dict(),'./checkpoint/net_{}.pth'.format(epoch))
        np.save("./checkpoint/loss_{}.npy".format(epoch),loss_lst)

        # evaluate function took too long to complete [skipped]
        # evaluate(net, DL_val, device=device)

    np.save("loss_lst.npy",loss_lst)
    print("loss list:",loss_lst)
    torch.save(net.state_dict(), './EgoHandTrainingModel.pth')
    print("model saved")
