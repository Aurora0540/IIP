import torch
import random
import numpy as np
import pandas as pd
import io
import matplotlib.pyplot as plt
from PIL import Image

# 配置
TRAIN_FUSION_PATH = "C:/Users/18320/Desktop/IIP/train_feature_fusion.pt"
TEST_FUSION_PATH  = "C:/Users/18320/Desktop/IIP/test_feature_fusion.pt"

TRAIN_PARQUET_PATH = "C:/Users/18320/Desktop/IIP/CIFAR100/train-00000-of-00001.parquet"
TEST_PARQUET_PATH  = "C:/Users/18320/Desktop/IIP/CIFAR100/test-00000-of-00001.parquet"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def denormalize(img):
    CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR100_STD = [0.2673, 0.2564, 0.2761]
    img = img.clone()
    for c in range(3):
        img[c] = img[c] * CIFAR100_STD[c] + CIFAR100_MEAN[c]
    return img.clamp(0, 1)

def load_image_from_data(img_data):
    if isinstance(img_data, dict) and "bytes" in img_data:
        return Image.open(io.BytesIO(img_data["bytes"])).convert("RGB")
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def compute_similarity(q_feat, gallery_feats, metric="cosine"):
    if metric == "cosine":
        q_feat = torch.nn.functional.normalize(q_feat.unsqueeze(0), dim=1)
        gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1)
        sim = torch.mm(q_feat, gallery_feats.t()).squeeze(0)
        return sim   # 越大越相似

    elif metric == "euclidean":
        dist = torch.cdist(q_feat.unsqueeze(0), gallery_feats).squeeze(0)
        return -dist  # 取负，方便 topk

    else:
        raise ValueError("metric must be 'cosine' or 'euclidean'")


def show_retrieval(query_img, gallery_imgs, query_label, idxs, gallery_labels, title):
    plt.figure(figsize=(15, 3))
    plt.subplot(1, len(gallery_imgs) + 1, 1)
    plt.imshow(query_img)
    plt.title(f"Query\nLabel: {query_label}")
    plt.axis("off")

    for i, img in enumerate(gallery_imgs):
        plt.subplot(1, len(gallery_imgs) + 1, i + 2)
        plt.imshow(img)
        plt.title(f"Top-{i+1}\nLabel: {gallery_labels[idxs[i]]}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def fusion_retrieve_topk(num_query,topk,DISTANCE,seed):


    # 1. 加载融合特征
    train_data = torch.load(TRAIN_FUSION_PATH)
    test_data  = torch.load(TEST_FUSION_PATH)

    train_feats = train_data["feat"]     # [N_train, D]
    train_labels = train_data["labels"]
    test_feats  = test_data["feat"]      # [N_test, D]
    test_labels = test_data["labels"]

    # 2. 加载原始数据（用于取图像）
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    test_df  = pd.read_parquet(TEST_PARQUET_PATH)

    # 3. 随机选 query（可复现）
    set_seed(seed)
    query_indices = random.sample(range(len(test_feats)), num_query)

    # 4. 检索 + 显示
    for q_idx in query_indices:
        q_feat = test_feats[q_idx]

        sim = compute_similarity(q_feat, train_feats, DISTANCE)
        vals, idxs = sim.topk(topk)

        query_img = load_image_from_data(test_df.iloc[q_idx]["img"])
        gallery_imgs = [
            load_image_from_data(train_df.iloc[i]["img"])
            for i in idxs.tolist()
        ]

        show_retrieval(
            query_img,
            gallery_imgs,
            test_labels[q_idx],
            idxs,
            train_labels,
            title=f"fusion_feature + {DISTANCE.upper()} | Query idx={q_idx}"
        )
        print(f"\nQuery label: {test_labels[q_idx]}")
        print("Top-5 matches:")
        for i in range(topk):
            print(
                f"  Rank {i + 1}: "
                f"Index={idxs[i].item()}, "
                f"Label={train_labels[idxs[i]].item()}, "
                f"Similarity or distance={vals[i].item():.4f}"
            )

if __name__ == "__main__":
    fusion_retrieve_topk(num_query=5, topk=10, DISTANCE="cosine", seed=2026)