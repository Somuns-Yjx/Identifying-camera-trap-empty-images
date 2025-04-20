import json
import os
import pandas as pd
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils
from ysrc.parameters_def import *


def md_model_init():
    # Initialize the MegaDetector model
    return pw_detection.MegaDetectorV6(device=GpuSet, pretrained=True, version="MDV6-yolov10-e")


def md_detect(detect_model, path, conf=0.01, batch_size=32, output_name_json=name_json_prd,
              output_name_csv=name_csv_prd):
    # Set output paths
    output_json_path = os.path.join(path, output_name_json)
    output_csv_path = os.path.join(path, output_name_csv)
    # Perform batch image detection
    results = detect_model.batch_image_detection(path, batch_size=batch_size, det_conf_thres=conf)
    # Save results in JSON format
    pw_utils.save_detection_timelapse_json(results, output_json_path,
                                           categories=detect_model.CLASS_NAMES,
                                           exclude_category_ids=[],
                                           exclude_file_path=path,
                                           info={"detector": "MegaDetectorV6"})
    # Read JSON data
    with open(output_json_path) as f:
        detection_data = json.load(f)
    images = detection_data[json_images]
    detections = []
    # Extract detection information
    for image in images:
        image_file = image[cn_file]
        for detection in image[json_detections]:
            detections.append({
                cn_file: image_file,
                json_category: '1',
                cn_conf: detection[json_conf],
                cn_bbox: detection[cn_bbox],
                cn_path_org: os.path.join(path)
            })
    detections_df = pd.DataFrame(detections)
    # Filter image files
    image_files = {file_name for file_name in os.listdir(path) if file_name.lower().endswith(format_images)}
    existing_files = set(detections_df[cn_file].values)
    # Add default rows for undetected images
    for file_name in image_files:
        if file_name not in existing_files:
            detections.append({
                cn_file: file_name,
                json_category: '0',
                cn_conf: 0,
                cn_bbox: '',
                cn_path_org: os.path.join(path)
            })
    detections_df = pd.DataFrame(detections)
    # Export to CSV
    detections_df.to_csv(output_csv_path, index=False)
    return output_json_path, output_csv_path
