import json
import os
import pandas as pd
from tqdm import tqdm  # Import the tqdm library
import shutil


def delete_crop_folder(folder_path):
    # Check if the folder exists
    crop_folder_path = os.path.join(folder_path, 'crop')
    if os.path.isdir(crop_folder_path):
        try:
            # Delete the 'crop' folder
            shutil.rmtree(crop_folder_path)
            print(f"Successfully deleted the folder: {crop_folder_path}")
        except Exception as e:
            print(f"Error while deleting folder {crop_folder_path}: {e}")
    else:
        print(f"No 'crop' folder found in {folder_path}")


def true_label_convert(json_path):
    # Define the file path
    output_dir = os.path.dirname(json_path)

    # Open and load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Progress bar
    file_list = ['info', 'categories', 'annotations']  # List of files to process
    data_list = [data['info'], data['categories'], data['annotations']]  # Corresponding extracted data

    # Wrap each file - saving operation with tqdm
    for file, data in tqdm(zip(file_list, data_list), total=3, desc="Processing files"):
        if file == 'info':
            df = pd.DataFrame([data])  # Convert to DataFrame
            df.to_csv(os.path.join(output_dir, 'info.csv'), index=False)
        elif file == 'categories':
            df = pd.DataFrame(data)  # Convert to DataFrame
            df.to_csv(os.path.join(output_dir, 'categories.csv'), index=False)
        elif file == 'annotations':
            df = pd.DataFrame(data)  # Convert to DataFrame
            df.to_csv(os.path.join(output_dir, 'annotations.csv'), index=False)

    print("CSV files have been created.")


def add_jpg(csv_file_dir_path):
    # Read the CSV file
    csv_path = os.path.join(csv_file_dir_path, 'annotations.csv')
    df = pd.read_csv(csv_path)

    # Add the ".jpg" suffix to each value in the image_id column
    df['image_id'] = df['image_id'].astype(str) + '.JPG'

    # Save the modified CSV file
    df.to_csv(csv_path, index=False)

    print("Successfully added the '.jpg' suffix to all image_id values.")