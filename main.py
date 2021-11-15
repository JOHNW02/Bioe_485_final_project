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


df = pd.read_csv("archive/trainLabels_cropped.csv")
image_names = df['image']
y = df['level']
print(Counter(y))

# C:\Users\jwei2\Bioe_485_final_project\archive\resized_train\resized_train\10_left.jpeg"
# images = []
# for image in tqdm(image_names):
#     img = cv2.imread(f'C:\\Users\\jwei2\\Bioe_485_final_project\\archive\\resized_train_cropped\\resized_train_cropped\\{image}.jpeg')
#     img = cv2.resize(img, (256,256))
#     if img is not None:
#         images.append(img)

# X = np.array(images)
# joblib.dump(X, "X")

# print(X.shape)


X = joblib.load("X")
X = X.reshape( (X.shape[0], 256*256*3) )
oversample = SMOTE()
X_over, y_over = oversample.fit_resample(X, y)
print("oversampled: ",Counter(y_over))
X_train, X_test, y_train, y_test = train_test_split(X_over, y_over, test_size=0.3)

clf = KNeighborsClassifier(n_neighbors=5)
clf.fit(X_train, y_train)

