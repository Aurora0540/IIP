import torch
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import io
from PIL import Image

# -----特征文件路径-----
GALLERY_FEAT_PATH = "C:/Users/18320/Desktop/IIP/gallery_features.pt"
QUERY_FEAT_PATH   = "C:/Users/18320/Desktop/IIP/query_features.pt"

TRAIN_PARQUET_PATH = "C:/Users/18320/Desktop/IIP/CIFAR100/train-00000-of-00001.parquet"
TEST_PARQUET_PATH  = "C:/Users/18320/Desktop/IIP/CIFAR100/test-00000-of-00001.parquet"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_image_from_data(img_data):
    if isinstance(img_data, dict) and 'bytes' in img_data:
        return Image.open(io.BytesIO(img_data['bytes'])).convert("RGB")
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def compute_similarity(q_feat, gallery_feats, metric="cosine"):
    if metric == "cosine":
        q_feat = torch.nn.functional.normalize(q_feat.unsqueeze(0), dim=1)
        gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1)
        sim = torch.mm(q_feat, gallery_feats.t()).squeeze(0)
        return sim  # 越大越相似

    elif metric == "euclidean":
        dist = torch.cdist(q_feat.unsqueeze(0), gallery_feats).squeeze(0)
        return -dist  # 取负数，方便 topk

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



# -----传统特征检索-----
def tradition_retrieve_topk(num_query,topk,DISTANCE,phase,seed):

    # 1. 加载特征
    gallery_data = torch.load(GALLERY_FEAT_PATH)
    query_data   = torch.load(QUERY_FEAT_PATH)

    gallery_labels = gallery_data["labels"]
    query_labels = query_data["labels"]
    if phase == "color":
        gallery_feats = torch.cat([gallery_data["color"]], dim=1)
        query_feats = torch.cat([query_data["color"]], dim=1)
    else:
        gallery_feats = torch.cat([gallery_data["sift"]], dim=1)
        query_feats = torch.cat([query_data["sift"]], dim=1)

    # 2. 加载原始数据（用于显示图像）
    train_df = pd.read_parquet(TRAIN_PARQUET_PATH)
    test_df  = pd.read_parquet(TEST_PARQUET_PATH)

    # 3. 随机选 query
    set_seed(seed)
    query_indices = random.sample(range(len(query_feats)), num_query)

    print('-' * 60)
    print("传统特征（颜色直方图、SIFT）检索:")
    print('-' * 60)
    for q_idx in query_indices:
        q_feat = query_feats[q_idx]

        # 4. 相似度计算
        sim = compute_similarity(q_feat, gallery_feats, DISTANCE)
        vals, idxs = sim.topk(topk)

        # 5. 取图像
        query_img = load_image_from_data(test_df.iloc[q_idx]["img"])
        gallery_imgs = [
            load_image_from_data(train_df.iloc[i]["img"])
            for i in idxs.tolist()
        ]

        # 6. 显示
        show_retrieval(
            query_img,
            gallery_imgs,
            query_labels[q_idx],
            idxs,
            gallery_labels,
            title=f"SIFT + {DISTANCE.upper()} | Query idx={q_idx}"
        )
        print(f"\nQuery label: {query_labels[q_idx]}")
        print("Top-5 matches:")
        for i in range(topk):
            print(
                f"  Rank {i + 1}: "
                f"Index={idxs[i].item()}, "
                f"Label={gallery_labels[idxs[i]].item()}, "
                f"Similarity or distance={vals[i].item():.4f}"
            )



if __name__ == '__main__':
    # 传统特征检索
    tradition_retrieve_topk(num_query=5, topk=10, DISTANCE="cosine", phase="color", seed=2026)
