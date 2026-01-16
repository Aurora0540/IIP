import pandas as pd
import numpy as np
import torch
import cv2
import io
import os
from PIL import Image
from sklearn.cluster import KMeans
from dataloader import ParquetImageDataset
from tqdm import tqdm


# 0. 配置路径与参数
TRAIN_PARQUET_PATH = r'C:/Users/18320/Desktop/IIP/CIFAR100/train-00000-of-00001.parquet'
TEST_PARQUET_PATH  = r'C:/Users/18320/Desktop/IIP/CIFAR100/test-00000-of-00001.parquet'
SAVE_DIR = r'C:/Users/18320/Desktop/IIP'
os.makedirs(SAVE_DIR, exist_ok=True)


def load_image_from_data(img_data):
    if isinstance(img_data, dict) and 'bytes' in img_data:
        return Image.open(io.BytesIO(img_data['bytes'])).convert('RGB')
    return Image.open(io.BytesIO(img_data)).convert('RGB')


def to_gray(img_obj):
    """安全的 RGB → Gray"""
    img = np.array(img_obj)
    if img.ndim == 2:
        return img
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    if img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported image shape: {img.shape}")



# 1. 特征提取类（颜色直方图 + SIFT）
class TraditionalFeatureExtractor:
    def __init__(self, sift_k=128):
        self.sift = cv2.SIFT_create()
        self.kmeans = KMeans(
            n_clusters=sift_k,
            n_init=10,
            random_state=42
        )
        self.sift_k = sift_k

    def extract_color_hist(self, img_obj):
        """HSV 颜色直方图"""
        cv_img = cv2.cvtColor(np.array(img_obj), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist(
            [cv_img], [0, 1], None,
            [12, 12], [0, 180, 0, 256]
        )
        cv2.normalize(hist, hist)
        return hist.flatten()

    def build_sift_vocab(self, img_objs):
        print("正在构建 SIFT 词典...")
        all_des = []
        for img in tqdm(img_objs[:500], desc="Collect SIFT descriptors"):
            gray = to_gray(img)
            _, des = self.sift.detectAndCompute(gray, None)
            if des is not None:
                all_des.append(des)

        all_des = np.vstack(all_des)
        self.kmeans.fit(all_des)
        print("SIFT 词典构建完成")

    def extract_sift_bovw(self, img_obj):
        gray = to_gray(img_obj)
        _, des = self.sift.detectAndCompute(gray, None)
        hist = np.zeros(self.sift_k, dtype=np.float32)

        if des is not None:
            preds = self.kmeans.predict(des)
            for p in preds:
                hist[p] += 1

        return hist / (hist.sum() + 1e-7)


# 2. 主流程
if __name__ == "__main__":

    # 划分数据集
    print("读取 train / test 数据集...")
    gallery_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    query_df = pd.read_parquet(TEST_PARQUET_PATH)

    print(f"Gallery 数量: {len(gallery_df)}, Query 数量: {len(query_df)}")

    extractor = TraditionalFeatureExtractor()

    for name, data_df in [("gallery", gallery_df), ("query", query_df)]:
        print(f"\n开始处理 {name} 集合...")

        img_objs = [load_image_from_data(d) for d in data_df['img']]
        labels = torch.tensor(data_df['coarse_label'].values)


        extractor.build_sift_vocab(img_objs)

        color_feats = []
        sift_feats = []

        for img in tqdm(img_objs, desc=f"Extract {name} features"):
            color_feats.append(extractor.extract_color_hist(img))
            sift_feats.append(extractor.extract_sift_bovw(img))

        color_feats = torch.tensor(color_feats, dtype=torch.float32)
        sift_feats = torch.tensor(sift_feats, dtype=torch.float32)

        # L2 归一化
        color_feats = torch.nn.functional.normalize(color_feats, dim=1)
        sift_feats = torch.nn.functional.normalize(sift_feats, dim=1)

        save_path = os.path.join(SAVE_DIR, f"{name}_features.pt")
        torch.save(
            {
                "color": color_feats,
                "sift": sift_feats,
                "labels": labels
            },
            save_path
        )

        print(f"{name} 特征已保存到: {save_path}")

    print(f"\n[完成] 所有特征已保存到目录: {SAVE_DIR}")
