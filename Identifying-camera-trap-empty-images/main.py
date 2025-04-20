import os
import traceback
from natsort import natsorted
from ysrc.md import md_model_init, md_detect
from ysrc.parameters_def import *
from yutils.preparation import delete_crop_folder
from yutils.mrg_repeat import csv_mrg_conf_part
from yutils.add_time import csv_add_time
from yutils.pair import image_find_pairs
from yutils.crop import image_crop_pairs
from yutils.sim_eff import cal_sim_efficientnet
from ytest.mrg_prd_ann_part import csv_mrg_pre_ann_part
from ytest.mrg_prd_ann_all import csv_mrg_part_to_all
from ytest.cal_indicators_md import cal_result_md_part, cal_result_md_all
from ytest.cal_indicators_sim import cal_result_sim_part, cal_result_sim_all


def func(folder_path, model):
    print(f"\nRunning func on folder: {folder_path}")
    prediction_file_path = os.path.join(folder_path, name_csv_prd)
    if not (os.path.exists(prediction_file_path)):
        try:
            md_detect(detect_model=model, path=folder_path, batch_size=128)
        except Exception as e:
            error_info = traceback.format_exc()
            with open('error_log.txt', 'a') as f:
                f.write(f"Folder: {folder_path}\nError: {str(e)}\nTraceback:\n{error_info}\n")

    delete_crop_folder(folder_path)
    csv_mrg_conf_part(folder_path)
    csv_add_time(folder_path)
    image_find_pairs(folder_path)
    image_crop_pairs(folder_path)
    cal_sim_efficientnet(folder_path)
    csv_mrg_pre_ann_part(folder_path)
    cal_result_sim_part(folder_path)
    cal_result_md_part(folder_path)


def main(base_path):
    model = md_model_init()

    for subdir, dirs, files in os.walk(base_path):
        dirs[:] = natsorted(dirs)
        files.sort()
        if any(file.lower().endswith(format_images) for file in files) and "crop" not in subdir.lower():
            func(subdir, model)

    csv_mrg_part_to_all(base_path, name_csv_eff_part, path_detect_root, name_csv_eff_all)
    csv_mrg_part_to_all(base_path, name_csv_md_part, path_detect_root, name_csv_md_all)
    cal_result_md_all(base_path)
    cal_result_sim_all(base_path, name_csv_eff_all, name_csv_result_sim_eff_all)


if __name__ == "__main__":
    main(path_detect_root)
