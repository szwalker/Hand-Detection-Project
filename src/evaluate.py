"""
evaluate.ipynb

Author: James (Jiaqi) Liu
"""

'''
from google.colab import drive
drive.mount('/content/drive',force_remount=True)
import os
os.chdir("/content/drive/My Drive/hw6")

# !pip uninstall pycocotools
!pip install cython
!pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
'''

import os
import numpy as np
from glob import glob
from PIL import Image,ImageDraw
from scipy.io import loadmat
from matplotlib.path import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import torch
import torch.nn as nn
import torchvision
import torch.utils.data as data
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import transforms as T
import utils
from engine import train_one_epoch,evaluate
import random

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

class EvalSet(data.Dataset):
    def __init__(self, img,label,img_fp): self.img,self.label,self.img_fp = img,label,img_fp
    def __len__(self): return len(self.img)
    def __getitem__(self, i): return self.img[i],self.label[i]
    def get_img_fp(self,i): return self.img_fp[i]

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train: transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

if __name__ == "__main__":
    # load model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    net = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    num_classes = 2
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    net_state_dict = torch.load("./EgoHandTrainingModel.pth")
    net.load_state_dict(net_state_dict)
    net.to(device)

    DS_eval = EgoHands('./egohands_data/_LABELLED_SAMPLES', get_transform(train=False))
    indices = [random.randint(0,len(DS_eval)-1) for _ in range(5)]
    img_arr = [DS_eval.get_img(_) for _ in indices]
    DS_eval = data.Subset(DS_eval, indices)

    img,label = [],[]
    for i,l in DS_eval:
      img.append(i)
      label.append(l)
    DS_eval = EvalSet(img,label,img_arr)
    DL_eval = data.DataLoader(DS_eval, batch_size=5, shuffle=False, collate_fn=utils.collate_fn,num_workers=12)

    evaluate(net,DL_eval,device)

    predictions = []
    net.to(device)
    net.eval()
    with torch.no_grad():
        image,targets = next(iter(DL_eval))
        print(image[0].shape)
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        predictions = net(image)   # Returns losses and detections
        final_img = []
        ans = []
        for i in range(len(image)):
            xy_lst = predictions[i]['boxes'].to(torch.device("cpu")).numpy()
            print(xy_lst)
            im = Image.open(img_arr[i])
            draw = ImageDraw.Draw(im)
            for box in xy_lst: draw.rectangle(box,outline='red',width=2)
            final_img.append(im)

    for i in range(len(final_img)): final_img[i].save('./img_output_{}'.format(i),'JPEG')

    # code for testing images (images not from EgoHands dataset)
    '''
    transform = transforms.Compose([transforms.ToTensor()])

    toy_test_imgs_path = sorted(glob(os.path.join('./toytest', "*.jpg")))
    print(toy_test_imgs_path)
    # toy_test_imgs = [Image.open(path).rotate(-90) for path in toy_test_imgs_path]
    toy_test_imgs = [Image.open(path) for path in toy_test_imgs_path]
    test_img_tensor_arr = [transform(img).to(device) for img in toy_test_imgs]
    net.eval()
    with torch.no_grad():
        prediction_arr = net(test_img_tensor_arr)
        for i in range(len(prediction_arr)):
            im = toy_test_imgs[i]
            draw = ImageDraw.Draw(im)
            xy_lst = prediction_arr[i]['boxes'].to(torch.device("cpu")).numpy()
            for box in xy_lst: draw.rectangle(box,outline='red',width=2)
    '''
