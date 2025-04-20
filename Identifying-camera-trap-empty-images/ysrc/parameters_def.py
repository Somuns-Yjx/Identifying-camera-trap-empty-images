import warnings
warnings.filterwarnings("ignore")

# Initialize the detection model
GpuSet = 'cuda:1'

# md detections keywords in json file
json_images = 'images'
json_detections = 'detections'
json_category = 'category'
json_conf = 'conf'

# md detect images format
format_images = '.jpg', '.jpeg', '.png'
format_time = "%Y/%m/%d %H:%M:%S"

# cm: column_name
cn_file = 'file'
cn_category = 'category'
cn_conf = 'confidence'
cn_bbox = 'bbox'
cn_path_org = 'path'
cn_time = 'time'

cn_path_img2 = 'path_file2'
cn_path_crop1 = 'path_crop_img1'
cn_path_crop2 = 'path_crop_img2'
cn_sim = 'similarity'

cn_ann_image_id = 'image_id'
cn_ann_category_id = 'category_id'
cn_pred = 'prediction'

# threshold
conf_thr_low = 0.01
conf_thr_high = 0.90

conf_thr_mid = 0.45
iou_thr_low_conf = 0.70  # iou
neighbors_low_conf = 3  # neighbor
day_delta = 7  # days
hour_delta = 3  # hours
time_thr = 10  # minutes

# file name
name_json_prd = "prediction.json"
name_csv_prd = "prediction.csv"
name_csv_prd_mrg = "prediction_merged.csv"

name_csv_md_part = "md_part.csv"
name_csv_md_all = "md_all.csv"
name_csv_result_md_part = "result_md_part.csv"
name_csv_result_md_all = "result_md_all.csv"

name_csv_eff_part = "eff_part.csv"
name_csv_eff_all = "eff_all.csv"
name_csv_result_eff_part = "result_eff_part.csv"
name_csv_result_sim_eff_all = "result_sim_eff_all.csv"

# dir name
name_dir_crop = "crop"
path_before_filename = "public"

path_detect_root = r'/data/users/yjx/datasets/ENO_S1/public'
path_annotation = r'/data/users/yjx/datasets/ENO_S1/annotations_linux.csv'

# path_detect_root = r'/data/users/yjx/datasets/CDB_S1/public'
# path_annotation = r'/data/users/yjx/datasets/CDB_S1/annotations_linux.csv'
