import os
import numpy as np
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras import backend as K
import torch

# Import the MultiResUNet model and utility functions
from pytorch.MultiResUNet import MultiResUnet, dice_coef, jacard, saveModel, evaluateModel, trainStep
from dataloading import load_data, split_data

# Define paths for data
IMAGE_DIR = 'data/imgs/'
MASK_DIR = 'data/masks/'

# Main training function
def main():
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load data
    X, Y = load_data(limit=10)  # Example: Load only 100 samples

    # Ensure Y has 4 channels (e.g., duplicate channels if necessary)
    if Y.shape[-1] == 1:
        Y = np.concatenate([Y] * 4, axis=-1)  # Duplicate the single channel to 4 channels

    # Split data into training and validation sets
    X_train, X_val, Y_train, Y_val = split_data(X, Y, validation=0.2)

    # Define the model
    model = MultiResUnet(input_channels=3, num_classes=4).to(device)  # Move model to GPU if available

    # Train the model
    trainStep(model, X_train, Y_train, X_val, Y_val, epochs=5, batch_size=4, device=device)

if __name__ == "__main__":
    main()