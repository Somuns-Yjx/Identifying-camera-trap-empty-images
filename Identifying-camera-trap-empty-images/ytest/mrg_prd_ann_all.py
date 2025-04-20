import os
import pandas as pd
from tqdm import tqdm


def csv_mrg_part_to_all(folder_path, csv_part_name, output_path, output_name):
    # Recursively find all CSV files with the specified name
    all_csv_files = [
        os.path.join(subdir, file)
        for subdir, _, files in os.walk(folder_path)
        for file in files if file.lower() == csv_part_name.lower()
    ]

    if not all_csv_files:
        print(f"Warning: No CSV files named '{csv_part_name}' found in {folder_path}.")
        return

    # Read CSV files in batches for efficiency
    dataframes = []
    for csv_path in tqdm(all_csv_files, desc=f"Merging {output_name}"):
        try:
            df = pd.read_csv(csv_path)
            if not df.empty:
                dataframes.append(df)
        except (pd.errors.EmptyDataError, FileNotFoundError) as e:
            print(f"Skipping {csv_path}: {e}")

    # Concatenate dataframes and save the merged result
    if dataframes:
        merged_df = pd.concat(dataframes, ignore_index=True)
        output_full_path = os.path.join(output_path, output_name)
        merged_df.to_csv(output_full_path, index=False)
    else:
        print("No valid data to merge.")
