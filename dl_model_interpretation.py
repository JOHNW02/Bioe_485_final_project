from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision.models import resnet50
import cv2
import numpy as np
import pandas as pd
import os
import joblib
from tqdm import tqdm
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from torch.utils.data import WeightedRandomSampler, DataLoader
from sklearn.metrics import roc_auc_score 
import joblib
import pickle
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataset import Dataset

random_seed=42
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)
torch.cuda.empty_cache()

# load original metadata
df = pd.read_csv("trainLabels_cropped.csv")
image_names = df['image'].tolist()
y = df['level']

subject0 = df.loc[ df['level']==0][0:10000]
subject1 = df.loc[ df['level']==1]
subject2 = df.loc[ df['level']==2]
subject3 = df.loc[ df['level']==3]
subject4 = df.loc[ df['level']==4]

# drop class 0 to 10000
df = pd.concat( (subject0, subject1, subject2, subject3, subject4), axis=0, ignore_index=True )
y = df['level'].tolist()
image_names = df['image']
index = np.arange(len(y))

# load preprocessed images
X = joblib.load("X")
print(X.shape)
X = np.swapaxes(X, 1, 3)
print(X.shape)

# make train, valid, heldout splits
train_val_indices, heldout_indices = train_test_split(index, test_size=0.3, random_state=random_seed, stratify=y)

# Load pretrained resnet 18. Here we only take 1 block of resnet 18 and add 2 more conv layers and 1 classification layer.
model = resnet18(pretrained=True)
model = torch.nn.Sequential(*(list(model.children())[0:5]),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False),
                nn.Conv2d(64, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(1, -1),
                nn.Linear(16, 5, bias=True)
                )
# Load model state
ckpt = torch.load("model_final.ckpt", map_location=device)
model.load_state_dict(ckpt['model_state'])
model = model.to(device)


heldout_dataset = Dataset(X, y, heldout_indices)
heldout_dataloader = DataLoader(heldout_dataset, batch_size=32, shuffle=False)

# Evaluate on heldout set
model.eval()
with torch.no_grad():
    y_probs_valid = torch.empty(0, 5).to(device)
    y_true_valid, y_pred_valid = [], []

    for i, batch in enumerate(heldout_dataloader):
        X_batch = batch["data"].to(device)
        y_batch = batch["label"].to(device)
        
        outputs = model(X_batch.float())

        _, predicted = torch.max(outputs.data, 1)
        y_pred_valid += predicted.cpu().numpy().tolist()
        y_probs_valid = torch.cat((y_probs_valid, outputs), 0)
        y_true_valid += y_batch.cpu().numpy().tolist()
    y_probs_valid = F.softmax(y_probs_valid, dim=1).cpu().numpy()
    y_true_valid = np.array(y_true_valid)

auc = roc_auc_score(y_true_valid, y_probs_valid, multi_class='ovr')
print("AUC: ", auc)
print(classification_report(y_true_valid, y_pred_valid))


X_heldout = X[heldout_indices]
y_pred_valid = np.array(y_pred_valid)


# Grad Cam
true_0_idx = []
true_1_idx = []
true_2_idx = []
true_3_idx = []
true_4_idx = []
for i in range(y_true_valid.shape[0]):
    if y_pred_valid[i] == 0 and y_true_valid[i] == 0:
        true_0_idx.append(i)
    if y_pred_valid[i] == 1 and y_true_valid[i] == 1:
        true_1_idx.append(i)
    if y_pred_valid[i] == 2 and y_true_valid[i] == 2:
        true_2_idx.append(i)
    if y_pred_valid[i] == 3 and y_true_valid[i] == 3:
        true_3_idx.append(i)
    if y_pred_valid[i] == 4 and y_true_valid[i] == 4:
        true_4_idx.append(i)

# We choose 5 images, that their predictions are correct.
to_produce_list = [true_0_idx[0], true_1_idx[0], true_2_idx[0], true_3_idx[0], true_4_idx[0] ]
heldout_img_idx = heldout_indices[to_produce_list]
img_names = image_names[heldout_img_idx]
print(img_names)
target_layers = [model[6]]

for i in range( len(to_produce_list) ):

    img_idx = to_produce_list[i]

    image_0 = np.float32(np.expand_dims(X_heldout[ img_idx ], axis=0) )

    input_tensor = torch.tensor(image_0).to(device).float()

    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)

    target_category = i

    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, target_category=target_category, aug_smooth=True)

    # In this example grayscale_cam has only one image in the batch:
    grayscale_cam = grayscale_cam[0, :]

    # Swap axes back to BGR
    to_test_image = np.swapaxes( np.float32(X_heldout[ img_idx ]), 0, 2)
    
    cam_image = show_cam_on_image(to_test_image, grayscale_cam, use_rgb=True)

    cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
    
    cv2.imwrite(f'class_{i}_cam.jpg', cam_image)