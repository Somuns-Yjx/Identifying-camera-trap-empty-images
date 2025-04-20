import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ysrc.parameters_def import *


def func_cal_matrix_md_all(csv_path, output_name):
    # Load CSV data
    df = pd.read_csv(csv_path)
    # Generate confidence thresholds from 0.01 to 1.00
    thresholds = np.arange(0.01, 1.01, 0.01)
    results = []

    # Calculate metrics for each threshold
    for conf_thr in tqdm(thresholds, desc=f"Calculating {output_name}"):
        pred = (df[cn_conf] >= conf_thr).astype(int)
        label = (df[cn_ann_category_id] != 0).astype(int)

        # Calculate confusion matrix components
        TP = ((pred == 1) & (label == 1)).sum()
        FP = ((pred == 1) & (label == 0)).sum()
        TN = ((pred == 0) & (label == 0)).sum()
        FN = ((pred == 0) & (label == 1)).sum()

        # Calculate performance metrics
        total = TP + TN + FP + FN
        accuracy = (TP + TN) / total if total else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        overall_error = (FP + FN) / total if total else 0
        omission_error = FN / (TP + FN) if (TP + FN) != 0 else 0
        commission_error = FP / (TP + FP) if (TP + FP) != 0 else 0
        removal = TN / (TN + FP) if (TN + FP) != 0 else 0  # New removal metric

        # Store results
        results.append({
            "conf_thr": round(conf_thr, 2),
            "TP": TP, "TN": TN, "FP": FP, "FN": FN,
            "accuracy": round(accuracy * 100, 2),
            "recall": round(recall * 100, 2),
            "precision": round(precision * 100, 2),
            "f1_score": round(f1_score * 100, 2),
            "overall_error": round(overall_error * 100, 2),
            "omission_error": round(omission_error * 100, 2),
            "commission_error": round(commission_error * 100, 2),
            "removal": round(removal * 100, 2)
        })

    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(os.path.dirname(csv_path), output_name), index=False)


def cal_result_md_part(folder_path):
    # Validate folder existence
    if not os.path.exists(folder_path):
        print(f"Error: The specified path does not exist: {folder_path}")
        return

    # Process part files
    for subdir, _, files in os.walk(folder_path):
        if name_csv_md_part in files:
            func_cal_matrix_md_all(os.path.join(subdir, name_csv_md_part), name_csv_result_md_part)


def cal_result_md_all(folder_path):
    # Process all files
    for root, _, files in os.walk(folder_path):
        if name_csv_md_all in files:
            func_cal_matrix_md_all(os.path.join(root, name_csv_md_all), name_csv_result_md_all)
