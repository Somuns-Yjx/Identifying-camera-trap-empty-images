import pandas as pd
from tqdm import tqdm
import os
from ysrc.parameters_def import *


def csv_mrg_conf_part(folder_path):
    # Find the CSV file in the folder
    entries = os.listdir(folder_path)
    if name_csv_prd in entries:
        csv_path = os.path.join(folder_path, name_csv_prd)
    else:
        return
    # Read the CSV into a DataFrame
    detections_df = pd.read_csv(csv_path)
    detections_df = detections_df.sort_values(by=detections_df.columns[0], ascending=True)
    detections_df[cn_conf] = pd.to_numeric(detections_df[cn_conf], errors='coerce')
    detections_df.sort_values(by=cn_file, ascending=True, inplace=True)
    merged_rows = []
    # Iterate groups and select rows
    for file_name, group in tqdm(detections_df.groupby(cn_file),
                                 desc="Merging repeat predictions",
                                 total=detections_df[cn_file].nunique()):
        if group[cn_conf].notna().any():
            max_row = group.loc[group[cn_conf].idxmax()]
            merged_rows.append(max_row)
        else:
            merged_rows.append(group.iloc[0])
    # Create new DataFrame from selected rows
    merged_df = pd.DataFrame(merged_rows)
    merged_df.fillna(0, inplace=True)
    # Save the result to a new CSV
    output_csv = os.path.join(os.path.dirname(csv_path), name_csv_prd_mrg)
    merged_df.to_csv(output_csv, index=False)
    merged_df.to_csv(os.path.join(os.path.dirname(csv_path), name_csv_md_part), index=False)
