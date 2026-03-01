import os
import cv2
import numpy as np
from tqdm import tqdm

def yolo_to_npz(img_dir, label_dir, output_dir):
    """
    Convert YOLO-style images and labels into .npz mask files with 4 channels.

    Args:
        img_dir (str): Directory containing the images.
        label_dir (str): Directory containing the YOLO-style label files.
        output_dir (str): Directory to save the .npz mask files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img_files = [f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
    label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

    img_files.sort()
    label_files.sort()

    for img_file, label_file in tqdm(zip(img_files, label_files), total=len(img_files)):
        # Read the image to get its dimensions
        img_path = os.path.join(img_dir, img_file)
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # Initialize a blank mask with 4 channels
        mask = np.zeros((height, width, 4), dtype=np.uint8)

        # Read the YOLO label file
        label_path = os.path.join(label_dir, label_file)
        with open(label_path, 'r') as f:
            for line in f:
                # Parse YOLO label (class_id, x_center, y_center, bbox_width, bbox_height)
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center = float(parts[1]) * width
                y_center = float(parts[2]) * height
                bbox_width = float(parts[3]) * width
                bbox_height = float(parts[4]) * height

                # Calculate bounding box coordinates
                x1 = int(x_center - bbox_width / 2)
                y1 = int(y_center - bbox_height / 2)
                x2 = int(x_center + bbox_width / 2)
                y2 = int(y_center + bbox_height / 2)

                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # Draw the bounding box on the corresponding channel
                if class_id < 4:  # Ensure class_id is within [0, 3]
                    mask[y1:y2, x1:x2, class_id] = 255

        # Save the mask as a .npz file
        output_path = os.path.join(output_dir, os.path.splitext(img_file)[0] + '.npz')
        np.savez_compressed(output_path, mask=mask)

if __name__ == "__main__":
    img_dir = "data/imgs"
    label_dir = "data/labels"
    output_dir = "data/masks"

    yolo_to_npz(img_dir, label_dir, output_dir)