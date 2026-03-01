import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_data(limit=None):
    img_files = next(os.walk('data/imgs'))[2]
    label_files = next(os.walk('data/masks'))[2]

    img_files.sort()
    label_files.sort()

    if limit:
        img_files = img_files[:limit]
        label_files = label_files[:limit]

    print(f"Number of image files: {len(img_files)}")
    print(f"Number of label files: {len(label_files)}")
    
    X = []
    Y = []

    for i in tqdm(img_files):
        # Load and preprocess image
        img = cv2.imread(os.path.join('data/imgs', i))
        img = img / 255.0  # Normalize image
        X.append(img)
    print(f"Finished loading images. {len(X)} images loaded.")
    for i in tqdm(label_files):
        # Load and preprocess mask from .npz file
        mask = np.load(os.path.join('data/masks', i))['mask']
        mask = mask / 255.0  # Normalize mask
        Y.append(mask)
    print(f"Finished loading masks. {len(Y)} masks loaded.")
    X = np.array(X, dtype='float32')  # Ensure data type is float32
    Y = np.array(Y, dtype='float32')
    print(f"X shape: {X.shape}")
    print(f"Y shape: {Y.shape}")    
    return X, Y

def split_data(X, Y, validation=0.1, random_state=42):
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation, random_state=random_state)
    print(f"Train set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    return X_train, X_val, Y_train, Y_val

if __name__ == "__main__": 
    X, Y = load_data(limit=100)  # Example: Load only 100 samples
    X_train, X_val, Y_train, Y_val = split_data(X, Y)