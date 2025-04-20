import contextlib
import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import cv2
import numpy as np
from tqdm import tqdm
from efficientnet_pytorch import EfficientNet

from ysrc.parameters_def import *

effb3_model = None


def get_feature_extractor():
    global effb3_model
    if effb3_model is None:
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            effb3_model = EfficientNet.from_pretrained('efficientnet-b3')
        effb3_model._avg_pooling = nn.Identity()
        effb3_model._fc = nn.Identity()
        effb3_model = effb3_model.to(GpuSet).eval()
    return effb3_model


def clahe_preprocess(image):
    image = np.array(image)
    yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(yuv[:, :, 0])
    enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return Image.fromarray(enhanced)


# Define the preprocessing method required by EfficientNet
preprocess = transforms.Compose([
    transforms.Lambda(clahe_preprocess),
    transforms.Resize(300),
    transforms.CenterCrop(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def load_image(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        return img
    except Exception as e:
        print(f"[ERROR] Failed to load image {image_path}: {e}")
        return None


# Extract EfficientNet features
def extract_features(image_path):
    global effb3_model
    effb3_model = get_feature_extractor()
    img = load_image(image_path)
    if img is None:
        return None
    input_tensor = preprocess(img).unsqueeze(0).to(GpuSet)
    with torch.no_grad():
        features = effb3_model.extract_features(input_tensor)
        features = nn.AdaptiveAvgPool2d(1)(features)
        features = features.flatten(start_dim=1)
    return features.squeeze(0).cpu().numpy()


def cosine_similarity(vec1, vec2):
    if vec1 is None or vec2 is None:
        return -1
    vec1 = torch.tensor(vec1, dtype=torch.float32).to(GpuSet).squeeze()
    vec2 = torch.tensor(vec2, dtype=torch.float32).to(GpuSet).squeeze()
    cos = nn.CosineSimilarity(dim=0)
    return cos(vec1, vec2).item()


# Calculate similarity using EfficientNet and update the original file
def cal_sim_efficientnet(folder_path):
    entries = os.listdir(folder_path)
    if name_csv_prd not in entries:
        print(f"[INFO] No prediction CSV found in {folder_path}, skipping.")
        return
    csv_mrg_path = os.path.join(folder_path, name_csv_prd_mrg)
    df = pd.read_csv(csv_mrg_path)
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Calculating similarity (EfficientNet)"):
        if conf_thr_low <= row[cn_conf] <= conf_thr_high:
            if row[cn_sim] != 0 and row[cn_sim] != 1:
                img1_path = row[cn_path_crop1]
                img2_path = row[cn_path_crop2]
                if pd.isna(img2_path) or img2_path == 'Null':
                    continue
                feat1 = extract_features(img1_path)
                feat2 = extract_features(img2_path)
                if feat1 is not None and feat2 is not None:
                    sim = cosine_similarity(feat1, feat2)
                    df.at[index, cn_sim] = sim
    output_path = os.path.join(os.path.dirname(csv_mrg_path), name_csv_eff_part)
    df.to_csv(output_path, index=False)
