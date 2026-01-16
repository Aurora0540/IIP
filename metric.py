import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# ----- 全局配置 -----
PATHS = {
    "gallery_trad": "C:/Users/18320/Desktop/IIP/gallery_features.pt",
    "query_trad": "C:/Users/18320/Desktop/IIP/query_features.pt",
    "cnn_train": "C:/Users/18320/Desktop/IIP/CNN_train_features.pt",
    "cnn_test": "C:/Users/18320/Desktop/IIP/CNN_test_features.pt",
    "fusion_train": "C:/Users/18320/Desktop/IIP/train_feature_fusion.pt",
    "fusion_test": "C:/Users/18320/Desktop/IIP/test_feature_fusion.pt",
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 100

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def compute_metrics_batch(query_feats, gallery_feats, query_labels, gallery_labels, k_values=[1, 5, 10],
                          metric="cosine"):

    # 归一化 (如果使用余弦相似度)
    if metric == "cosine":
        query_feats = torch.nn.functional.normalize(query_feats, dim=1)
        gallery_feats = torch.nn.functional.normalize(gallery_feats, dim=1)

    num_queries = query_feats.shape[0]

    # 初始化统计器
    # precision_sums[k] 存储所有 query 在 top-k 的累积 precision
    precision_sums = {k: 0.0 for k in k_values}
    recall_sums = {k: 0.0 for k in k_values}

    max_k = max(k_values)

    # 预先计算 Gallery 中每个类别的总样本数，用于 Recall 分母
    gallery_labels_cpu = gallery_labels.cpu().numpy()
    unique_labels, counts = np.unique(gallery_labels_cpu, return_counts=True)
    label_counts = dict(zip(unique_labels, counts))

    # 分批处理 Query，避免生成巨大的 [N_query, N_gallery] 矩阵
    set_seed(42)
    for i in tqdm(range(0, num_queries, BATCH_SIZE), desc=f"Calculating Metrics ({metric})"):
        batch_q = query_feats[i: i + BATCH_SIZE].to(DEVICE)
        batch_labels = query_labels[i: i + BATCH_SIZE].to(DEVICE)

        # 计算距离矩阵
        if metric == "cosine":
            # sim: [batch, N_gallery]
            sim_matrix = torch.mm(batch_q, gallery_feats.t())
            # topk, largest=True for cosine
            _, topk_indices = sim_matrix.topk(max_k, dim=1, largest=True)
        else:
            # Euclidean
            dist_matrix = torch.cdist(batch_q, gallery_feats)
            # topk, largest=False for distance
            _, topk_indices = dist_matrix.topk(max_k, dim=1, largest=False)

        # 转换为标签矩阵 [batch, max_k]
        retrieved_labels = gallery_labels[topk_indices]

        # 计算匹配矩阵 (Broadcasting)
        matches = retrieved_labels == batch_labels.unsqueeze(1)  # [batch, max_k] Boolean

        # 遍历每个 K 值计算指标
        for k in k_values:
            # 截取前 k 列
            matches_k = matches[:, :k].float()

            # --- Precision@K ---
            # 每个 query 匹配到的数量 / k
            num_correct_k = matches_k.sum(dim=1)
            p_k = num_correct_k / k
            precision_sums[k] += p_k.sum().item()

            # --- Recall@K ---
            # 每个 query 匹配到的数量 / 该类别在数据库中的总数
            # 需要逐个查表获取分母
            current_batch_labels = batch_labels.cpu().numpy()
            denominators = np.array([label_counts.get(l, 1) for l in current_batch_labels])
            denominators = torch.from_numpy(denominators).float().to(DEVICE)

            r_k = num_correct_k / denominators
            recall_sums[k] += r_k.sum().item()

    # 计算平均值
    results = {}
    for k in k_values:
        results[f"P@{k}"] = precision_sums[k] / num_queries
        results[f"R@{k}"] = recall_sums[k] / num_queries

    return results


def eval_traditional():
    print("\n" + "=" * 40)
    print("Evaluating Traditional Features (Color)...")
    print("=" * 40)

    # 加载数据
    g_data = torch.load(PATHS["gallery_trad"])
    q_data = torch.load(PATHS["query_trad"])

    g_feats = g_data["color"].to(DEVICE)
    g_labels = g_data["labels"].to(DEVICE)
    q_feats = q_data["color"].to(DEVICE)
    q_labels = q_data["labels"].to(DEVICE)

    return compute_metrics_batch(q_feats, g_feats, q_labels, g_labels, metric="cosine")


def eval_cnn():
    print("\n" + "=" * 40)
    print("Evaluating CNN Features (ResNet50)...")
    print("=" * 40)

    # 1. 加载 Gallery 特征 (已保存)
    g_data = torch.load(PATHS["cnn_train"])
    g_feats = g_data["features"].to(DEVICE)
    g_labels = g_data["labels"].to(DEVICE)

    q_data = torch.load(PATHS["cnn_test"])
    q_feats = q_data["features"].to(DEVICE)
    q_labels = q_data["labels"].to(DEVICE)

    return compute_metrics_batch(q_feats, g_feats, q_labels, g_labels, metric="cosine")


def eval_fusion():
    print("\n" + "=" * 40)
    print("Evaluating Fusion Features...")
    print("=" * 40)

    train_data = torch.load(PATHS["fusion_train"])
    test_data = torch.load(PATHS["fusion_test"])

    g_feats = train_data["feat"].to(DEVICE)
    g_labels = train_data["labels"].to(DEVICE)
    q_feats = test_data["feat"].to(DEVICE)
    q_labels = test_data["labels"].to(DEVICE)

    return compute_metrics_batch(q_feats, g_feats, q_labels, g_labels, metric="cosine")


def plot_results(all_results):
    k_values = [1, 5, 10]

    # 准备数据绘图
    methods = list(all_results.keys())

    # Plot Precision
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for method in methods:
        p_vals = [all_results[method][f"P@{k}"] for k in k_values]
        plt.plot(k_values, p_vals, marker='o', label=method)
    plt.title("Precision@K Comparison")
    plt.xlabel("K")
    plt.ylabel("Precision")
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Plot Recall
    plt.subplot(1, 2, 2)
    for method in methods:
        r_vals = [all_results[method][f"R@{k}"] for k in k_values]
        plt.plot(k_values, r_vals, marker='s', label=method)
    plt.title("Recall@K Comparison")
    plt.xlabel("K")
    plt.ylabel("Recall")
    plt.xticks(k_values)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.show()


def main():
    # 存储所有结果
    final_metrics = {}

    # 1. 传统特征评估
    try:
        final_metrics["Traditional (Color)"] = eval_traditional()
    except Exception as e:
        print(f"Skipping Traditional: {e}")

    # 2. CNN特征评估
    try:
        final_metrics["CNN (ResNet50)"] = eval_cnn()
    except Exception as e:
        print(f"Skipping CNN: {e}")

    # 3. 融合特征评估
    try:
        final_metrics["Fusion"] = eval_fusion()
    except Exception as e:
        print(f"Skipping Fusion: {e}")

    # 4. 展示表格
    print("\n" + "=" * 80)
    print("FINAL EXPERIMENTAL RESULTS")
    print("=" * 80)

    df = pd.DataFrame(final_metrics).T
    # 调整列顺序
    cols = sorted(df.columns, key=lambda x: (x[0], int(x.split('@')[1])))
    df = df[cols]

    print(df)
    print("=" * 80)

    # 5. 可视化对比
    plot_results(final_metrics)


if __name__ == "__main__":
    main()