import os
import ast
import pandas as pd
from datetime import datetime, timedelta
from tqdm import tqdm
from ysrc.parameters_def import *


def image_initial_filter(folder_path):
    # Check if the merged CSV file exists
    entries = os.listdir(folder_path)
    if name_csv_prd_mrg not in entries:
        return
    # Read the merged CSV file
    csv_mrg_path = os.path.join(folder_path, name_csv_prd_mrg)
    df = pd.read_csv(csv_mrg_path)
    for index_base, row in tqdm(df.iterrows(), total=len(df), desc="Initial filtering"):
        img1_conf = row[cn_conf]
        img1_bbox = ast.literal_eval(row[cn_bbox])
        if img1_conf == 0 or img1_conf > conf_thr_high:
            continue
        if cal_iou_ratio(img1_bbox):
            continue
        df_event = get_event_img(df, index_base)
        if conf_thr_low <= img1_conf < conf_thr_mid:
            for i in range(min(neighbors_low_conf, len(df_event))):
                row_img2_low = df_event.iloc[i]
                if conf_thr_low <= row_img2_low[cn_conf] <= conf_thr_mid:
                    if cal_iou_for_low_conf(img1_bbox, ast.literal_eval(row_img2_low[cn_bbox])):
                        df.at[index_base, cn_conf] = 0
                        df.at[row_img2_low.name, cn_conf] = 0
                        break
    df.to_csv(csv_mrg_path, index=False)


def image_find_pairs(folder_path):
    image_initial_filter(folder_path)

    entries = os.listdir(folder_path)
    if name_csv_prd not in entries:
        return

    csv_mrg_path = os.path.join(folder_path, name_csv_prd_mrg)
    df = pd.read_csv(csv_mrg_path)

    crop_folder_path = os.path.join(folder_path, name_dir_crop)
    os.makedirs(crop_folder_path, exist_ok=True)

    df[cn_path_img2] = ''
    df[cn_path_crop1] = ''
    df[cn_path_crop2] = ''
    df[cn_sim] = ''

    for index_base, row in tqdm(df.iterrows(), total=len(df), desc="Pairing images"):
        img1_name = row[cn_file]
        img1_conf = row[cn_conf]
        img1_bbox = ast.literal_eval(row[cn_bbox])
        img1_crop_path = os.path.join(crop_folder_path, row[cn_file])
        in_event = 0

        if img1_conf == 0:  
            continue
        if img1_conf > conf_thr_high:
            continue
        if cal_iou_ratio(img1_bbox):
            continue
        df_event = get_event_img(df, index_base)

        if conf_thr_low <= img1_conf < conf_thr_high:
            for index_low_conf_find_emp, img2_row in df_event.iterrows():
                if img2_row[cn_conf] == 0:
                    img2_path = os.path.join(img2_row[cn_path_org], img2_row[cn_file])
                    img2_crop_name = f"{os.path.splitext(img1_name)[0]}_context_{os.path.basename(img2_path)}"
                    df.at[index_base, cn_path_img2] = img2_path
                    df.at[index_base, cn_path_crop1] = img1_crop_path
                    df.at[index_base, cn_path_crop2] = os.path.join(crop_folder_path, img2_crop_name)
                    in_event = 1
                    break
            if in_event:
                continue

            out_event_row = find_empty_out_of_event(df, index_base)
            if out_event_row is not None:
                img2_path = os.path.join(out_event_row[cn_path_org], out_event_row[cn_file])
                img2_crop_name = f"{os.path.splitext(img1_name)[0]}_context_{os.path.basename(img2_path)}"
                df.at[index_base, cn_path_img2] = os.path.join(out_event_row[cn_path_org],
                                                               out_event_row[cn_file])
                df.at[index_base, cn_path_crop1] = img1_crop_path
                df.at[index_base, cn_path_crop2] = os.path.join(crop_folder_path, img2_crop_name)

    df.to_csv(csv_mrg_path, index=False)


def get_event_img(df, img1_index):
    # Create a new DataFrame to store event images
    df1 = pd.DataFrame(columns=df.columns)
    img1_time = datetime.strptime(df.iloc[img1_index][cn_time], format_time)
    offset = 1
    stop_up = stop_down = False
    # Search for event images
    while not (stop_up and stop_down):
        found_rows = []
        if not stop_up:
            upper_index = img1_index - offset
            if upper_index >= 0:
                img2_time = datetime.strptime(df.iloc[upper_index][cn_time], format_time)
                if abs(img1_time - img2_time) <= timedelta(minutes=time_thr):
                    found_rows.append(upper_index)
                else:
                    stop_up = True
            else:
                stop_up = True
        if not stop_down:
            lower_index = img1_index + offset
            if lower_index < len(df):
                img2_time = datetime.strptime(df.iloc[lower_index][cn_time], format_time)
                if abs(img2_time - img1_time) <= timedelta(minutes=time_thr):
                    found_rows.append(lower_index)
                else:
                    stop_down = True
            else:
                stop_down = True
        for idx in sorted(found_rows):
            df1 = pd.concat([df1, df.iloc[[idx]]], ignore_index=False)
        offset += 1
    return df1


def find_empty_out_of_event(df, base_index):
    base_time = datetime.strptime(df.iloc[base_index][cn_time], format_time)
    offset = 1
    up_stop = down_stop = False
    # Search for empty image out of the event
    while not (up_stop and down_stop):
        if not up_stop:
            up_index = base_index - offset
            if up_index >= 0:
                row_up = df.iloc[up_index]
                up_time = datetime.strptime(row_up[cn_time], format_time)
                if abs((base_time - up_time).days) <= day_delta and row_up[cn_conf] == 0:
                    if abs((base_time - up_time).total_seconds() / 3600) < hour_delta:
                        return row_up
                else:
                    up_stop = True
            else:
                up_stop = True
        if not down_stop:
            down_index = base_index + offset
            if down_index < len(df):
                row_down = df.iloc[down_index]
                down_time = datetime.strptime(row_down[cn_time], format_time)
                if abs((down_time - base_time).days) <= day_delta and row_down[cn_conf] == 0:
                    if abs((down_time - base_time).total_seconds() / 3600) < hour_delta:
                        return row_down
                else:
                    down_stop = True
            else:
                down_stop = True
        offset += 1
    return None


def cal_iou_ratio(img1_bbox):
    x1, y1, x2, y2 = img1_bbox
    return 1 if (x2 - x1) * (y2 - y1) > 0.80 else 0


def cal_iou_for_low_conf(bbox1, bbox2):
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    area1 = w1 * h1
    area2 = w2 * h2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    ratio1 = inter_area / area1 if area1 > 0 else 0
    ratio2 = inter_area / area2 if area2 > 0 else 0
    return ratio1 > iou_thr_low_conf and ratio2 > iou_thr_low_conf
