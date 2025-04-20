import os
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from PIL import Image
from PIL.ExifTags import TAGS
from ysrc.parameters_def import *


def get_image_datetime(path):
    try:
        # Read EXIF data from image
        image = Image.open(path)
        exif_data = image._getexif()

        if exif_data:
            # Check for DateTime tag
            for tag, value in exif_data.items():
                if TAGS.get(tag) == 'DateTime':
                    return datetime.strptime(value, "%Y:%m:%d %H:%M:%S").strftime("%Y/%m/%d %H:%M:%S")

        # Fallback to file modification time
        return datetime.fromtimestamp(os.path.getmtime(path)).strftime("%Y/%m/%d %H:%M:%S")
    except Exception:
        return None


def csv_add_time(folder_path):
    entries = os.listdir(folder_path)
    if name_csv_prd_mrg not in entries:
        return

    # Load CSV file
    csv_path = os.path.join(folder_path, name_csv_prd_mrg)
    df = pd.read_csv(csv_path)

    # Process each image entry
    for index, row in tqdm(df.iterrows(), desc="Adding time info", total=len(df)):
        image_path = os.path.join(row[cn_path_org], row[cn_file])

        if os.path.exists(image_path):
            capture_time = get_image_datetime(image_path)
            if capture_time:
                df.at[index, cn_time] = capture_time

    # Save updated CSV
    df.to_csv(csv_path, index=False)