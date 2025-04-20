import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from ysrc.parameters_def import *


def func_cal_result_sim(csv_path, output_name):
    # Load CSV data
    df = pd.read_csv(csv_path)
    # Generate similarity thresholds from 0.01 to 1.00
    thresholds = np.arange(0.01, 1.01, 0.01)
    results = []

    with tqdm(total=len(thresholds), desc=f"Processing {os.path.basename(csv_path)}") as pbar:
        for similarity_thr in thresholds:
            # Calculate predictions based on confidence and similarity thresholds
            df[cn_pred] = np.where(
                df[cn_sim].isna(),
                np.where(df[cn_conf] < conf_thr_low, 0, 1),
                np.where(df[cn_sim] == 1, 0, np.where(df[cn_sim] > similarity_thr, 0, 1))
            )

            # Calculate confusion matrix components
            label = (df[cn_ann_category_id] != 0).astype(int)
            pred = df[cn_pred]

            TP = ((pred == 1) & (label == 1)).sum()
            FP = ((pred == 1) & (label == 0)).sum()
            TN = ((pred == 0) & (label == 0)).sum()
            FN = ((pred == 0) & (label == 1)).sum()

            # Calculate performance metrics including new removal metric
            total = TP + TN + FP + FN
            accuracy = (TP + TN) / total if total else 0
            recall = TP / (TP + FN) if (TP + FN) != 0 else 0
            precision = TP / (TP + FP) if (TP + FP) != 0 else 0
            f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
            overall_error = (FP + FN) / total if total else 0
            omission_error = FN / (TP + FN) if (TP + FN) != 0 else 0
            commission_error = FP / (TP + FP) if (TP + FP) != 0 else 0
            removal = TN / (TN + FP) if (TN + FP) != 0 else 0

            # Store results
            results.append({
                "similarity_thr": round(similarity_thr, 2),
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

            pbar.update(1)

    # Save updated CSV and results
    df.to_csv(csv_path, index=False)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(os.path.dirname(csv_path), output_name), index=False)


def cal_result_sim_part(folder_path):
    # Process specific CSV files in target folder
    files_to_process = [
        (name_csv_eff_part, name_csv_result_eff_part)
    ]

    for file_name, result_name in files_to_process:
        file_path = os.path.join(folder_path, file_name)
        if os.path.exists(file_path):
            func_cal_result_sim(file_path, result_name)
        else:
            print(f"File not found: {file_path}")


def cal_result_sim_all(root_path, file_name, output_name):
    # Process merged CSV file in root path
    file_path = next((os.path.join(root_path, f) for f in os.listdir(root_path) if f.lower() == file_name.lower()),
                     None)
    if file_path:
        func_cal_result_sim(file_path, output_name)
    else:
        print(f"File {file_name} not found in {root_path}.")
