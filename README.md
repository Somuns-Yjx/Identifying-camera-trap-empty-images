# **A stepwise method for identifying camera trap empty images combining MegaDetector and context image similarity**

This experiment uses MegaDetector for the initial screening of camera trap images. If you haven't installed MegaDetector yet, please visit the website [Microsoft AI for Earth: MegaDetector](https://github.com/microsoft/CameraTraps) get details first.

## Dataset Description

This experiment takes the **Snapshot Enonkishu** and **Snapshot Camdeboo** as example datasets, and the task is identifying camera trap empty images.

## Folder Structure

This project contains the following folders and files:

### 1. `ysrc/`

This folder contains the relevant code for the source:

- `md.py`: The code used to detect datasets by the MD model.
- `parameters_def.py`: The code used to define the global parameters in this project.

### 2. `yutils/`

This folder contains the relevant code for the preparation and algorithm:

- `preparation.py`: The code is used to process ground-truth labels and remove intermediate files.
- `mrg_repeat.py`: The code used to select the highest object detecting box in each image.
- `add_time.py`:  The code is used to add the image capture time.
- `pair.py`: The code is used to construct the image pair.
- `crop.py`: The code is used to crop image pairs.
- `sim_eff.py`: The code used to calculate similarity of image pairs.

### 3. `ytest/`

This folder contains the relevant code for calculating evaluation metrics:

- `mrg_prd_ann_part.py`:  The code is used to merge ground-truth labels and intermediate files.
- `mrg_prd_ann_all.py`: The code is used to merge ground-truth labels and final files..
- `cal_indicators_md.py`: The code is used to calculate evaluation metrics of the MD model under different confidence thresholds.
- `cal_indicators_sim.py`: The code is used to calculate evaluation metrics of propose method.

## Usage Instructions

### 1. Install Dependencies

Please ensure that **MegaDetector** and the relevant Python dependencies have been installed. 

### 2. path

Please change the path of dataset and annotation files

```python
# root path of dataset
path_detect_root = r''
# path of annotation files
path_annotation = r''
```

### 3. Run

If all prerequisites are met, executing the following code will automatically run the program and perform the evaluation.

```python
python main.py
```

