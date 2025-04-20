import os
import cv2
import ast
import pandas as pd
from tqdm import tqdm
# Importing parameters from a custom module
from ysrc.parameters_def import *


def image_crop_pairs(folder_path):
    # Define the path for the crop folder
    crop_folder_path = os.path.join(folder_path, "crop")

    # Create the crop folder if it doesn't exist
    if not os.path.exists(crop_folder_path):
        os.makedirs(crop_folder_path)

    # Get the list of entries in the folder
    entries = os.listdir(folder_path)
    # Check if the specified CSV file exists
    if name_csv_prd in entries:
        csv_mrg_path = os.path.join(folder_path, name_csv_prd_mrg)
    else:
        return
    # Read the CSV file into a DataFrame
    df = pd.read_csv(csv_mrg_path)
    # Iterate through each row in the DataFrame with a progress bar
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Cropping images"):

        # Get the paths of the original and second images
        img1_path = os.path.join(row[cn_path_org], row[cn_file])
        img2_path = row[cn_path_img2]

        # Check if the paths are valid strings and img2_path is not 'Null'
        if isinstance(img1_path, str) and isinstance(img2_path, str) and img2_path != 'Null':
            if img1_path and not pd.isna(img1_path):
                # Get the bounding box string from the row
                bbox_str = row.get(cn_bbox, None)
                # Get the paths for the cropped images
                pair1_path = row[cn_path_crop1]
                pair2_path = row[cn_path_crop2]
                # Convert the bounding box string to a tuple
                bbox = ast.literal_eval(bbox_str)
                # Check if the paths for cropped images are valid
                if not pd.isna(pair1_path) and not pd.isna(pair2_path):
                    # Crop and save the first image
                    crop_and_save(img1_path, pair1_path, bbox)
                    # Crop and save the second image
                    crop_and_save(img2_path, pair2_path, bbox)


def crop_and_save(org_path, crop_path, bbox):
    # Read the original image
    image = cv2.imread(org_path)
    # Get the height and width of the image
    height, width = image.shape[:2]
    # Unpack the bounding box coordinates
    x, y, w, h = bbox

    # Convert relative coordinates to absolute coordinates
    x = int(x * width)
    y = int(y * height)
    w = int(w * width)
    h = int(h * height)

    # Crop the image according to the bounding box
    cropped_image = image[y:y + h, x:x + w]

    # Check if the crop path is valid
    if not crop_path or not isinstance(crop_path, str):
        print(f"Invalid crop path: {crop_path}")
    else:
        # Save the cropped image
        cv2.imwrite(crop_path, cropped_image)
