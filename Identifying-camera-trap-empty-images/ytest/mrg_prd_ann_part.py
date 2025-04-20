import os
import pandas as pd
from tqdm import tqdm
from ysrc.parameters_def import *


def merge_md_and_annotation_part(prd_csv_path, ann_csv_path):
    # Validate input files
    if not os.path.exists(prd_csv_path) or os.path.getsize(prd_csv_path) == 0:
        print(f"Error: Invalid or empty file: {prd_csv_path}")
        return

    # Read CSV files
    try:
        df_prd = pd.read_csv(prd_csv_path)
        df_anno = pd.read_csv(ann_csv_path)
    except Exception as e:
        print(f"Error reading files: {e}")
        return

    # Check required global variable
    if 'path_before_filename' not in globals():
        print("Error: 'path_before_filename' is not defined.")
        return

    # Prepare base path and annotation index
    base_path = f"{path_before_filename}{os.path.sep}"
    df_anno_indexed = df_anno.set_index(cn_ann_image_id, drop=False)
    merged_rows = []

    # Process each prediction row
    for _, row_prd in tqdm(df_prd.iterrows(), total=len(df_prd), desc=f"Merging {os.path.basename(prd_csv_path)}"):
        raw_path = row_prd.get(cn_path_org, '')
        file_name = row_prd.get(cn_file, '')

        # Find base path in raw path
        idx_public = raw_path.lower().find(base_path.lower())
        if idx_public == -1:
            print(f"Warning: {base_path} not found in path => {raw_path}")
            continue

        # Construct new path for annotation lookup
        subdir_after_public = raw_path[idx_public + len(base_path):].lstrip(os.path.sep)
        new_str = os.path.join(subdir_after_public, file_name)
        # new_str = file_name

        # Merge with annotation if found
        if new_str in df_anno_indexed.index:
            row_anno = df_anno_indexed.loc[new_str]
            if isinstance(row_anno, pd.DataFrame):
                row_anno = row_anno.iloc[0]

            merged_dict = dict(row_prd)
            merged_dict[cn_ann_category_id] = row_anno[cn_ann_category_id]
            merged_rows.append(merged_dict)
        else:
            print(f"Warning: '{new_str}' not found in annotation. Skipping...")

    # Save merged results
    if merged_rows:
        df_merged = pd.DataFrame(merged_rows)
        df_merged.to_csv(prd_csv_path, index=False)
    else:
        print("No matching annotations found.")


def csv_mrg_pre_ann_part(folder_path):
    # Validate target folder
    if not os.path.exists(folder_path):
        print(f"Error: The specified path does not exist: {folder_path}")
        return

    # Process all relevant files
    file_names = [
        name_csv_eff_part,
        name_csv_md_part
    ]

    for file_name in file_names:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            merge_md_and_annotation_part(file_path, path_annotation)
        else:
            print(f"File not found: {file_path}")
