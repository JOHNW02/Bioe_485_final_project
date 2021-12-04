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
# set seed
random_seed=42

# load original metadata
df = pd.read_csv("trainLabels_cropped.csv")
image_names = df['image']
y = df['level']


subject0 = df.loc[ df['level']==0][0:10000]
subject1 = df.loc[ df['level']==1]
subject2 = df.loc[ df['level']==2]
subject3 = df.loc[ df['level']==3]
subject4 = df.loc[ df['level']==4]

# drop class 0 to 10000
df = pd.concat( (subject0, subject1, subject2, subject3, subject4), axis=0, ignore_index=True )
y = df['level']
image_names = df['image']
index = np.arange(len(y))

# load preprocessed images
X = joblib.load("X")

X = np.swapaxes(X, 1, 3)
print("image shape: ", X.shape)

# make train, valid, heldout splits
train_val_indices, heldout_indices = train_test_split(index, test_size=0.3, random_state=random_seed, stratify=y)

train_indices, valid_indices = train_test_split(train_val_indices, test_size=0.2, random_state=random_seed, stratify=y[train_val_indices])

labels_unique, counts = np.unique(y[train_val_indices], return_counts=True)
class_weights = np.array([1.0 / x for x in counts])
# provide weights for samples in the training set only
sample_weights = class_weights[y[train_indices]]
# sampler needs to come up with training set size number of samples
weighted_sampler = WeightedRandomSampler( weights=sample_weights, num_samples=len(train_indices), replacement=True)

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

# Set Dataset and dataloader
train_dataset = Dataset(X, y, train_indices)
train_dataloader = DataLoader(train_dataset, batch_size=32, sampler=weighted_sampler)
test_dataset = Dataset(X, y, valid_indices)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Number of epochs
epochs = 100

# Set device
device = torch.device('cuda:0')

# We are using cross entropy loss as loss function
loss_function = nn.CrossEntropyLoss()

# Using adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
model = model.to(device)

# Trainning
train_loss_history, valid_loss_history = [], []

lowest_loss = 9999

for epoch in range(epochs):
  model.train()
  train_loss = []
  for batch_idx, batch in enumerate(train_dataloader):
    X_batch = batch["data"].to(device)
    y_batch = batch["label"].to(device)
    optimizer.zero_grad()
    
    outputs = model(X_batch.float())
    loss = loss_function(outputs, y_batch)
    train_loss.append(loss.item())
    loss.backward()
    optimizer.step()

  model.eval()
  with torch.no_grad():
    valid_loss = []
    y_probs_valid = torch.empty(0, 5).to(device)
    y_true_valid, y_pred_valid = [], []

    for i, batch in enumerate(test_dataloader):
      X_batch = batch["data"].to(device)
      y_batch = batch["label"].to(device)
      
      outputs = model(X_batch.float())
      loss = loss_function(outputs, y_batch)
      valid_loss.append(loss.item())

      _, predicted = torch.max(outputs.data, 1)
      y_pred_valid += predicted.cpu().numpy().tolist()
      y_probs_valid = torch.cat((y_probs_valid, outputs), 0)
      y_true_valid += y_batch.cpu().numpy().tolist()
  y_probs_valid = F.softmax(y_probs_valid, dim=1).cpu().numpy()
  y_true_valid = np.array(y_true_valid)

  train_loss_history.append(np.mean(train_loss))
  valid_loss_history.append(np.mean(valid_loss))
  
  auc = roc_auc_score(y_true_valid, y_probs_valid, multi_class='ovr')
  print(f"Epoch {epoch} train loss: {train_loss_history[-1]} ", f"test loss: { valid_loss_history[-1] }")
  print("AUC: ", auc)
  print(classification_report(y_true_valid, y_pred_valid))
  
  # Save model
  state = {
      'model_description': str(model),
      'model_state': model.state_dict(),
      'optimizer': optimizer.state_dict()
  }

 
  torch.save(state, f"model_{epoch}.ckpt")
