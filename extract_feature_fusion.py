import torch
import os

# 配置路径
TRAIN_DEEP_PATH = "C:/Users/18320/Desktop/IIP/CNN_train_features.pt"
TEST_DEEP_PATH  = "C:/Users/18320/Desktop/IIP/CNN_test_features.pt"

TRAIN_TRAD_PATH = "C:/Users/18320/Desktop/IIP/gallery_features.pt"
TEST_TRAD_PATH  = "C:/Users/18320/Desktop/IIP/query_features.pt"

SAVE_DIR = "C:/Users/18320/Desktop/IIP/"
os.makedirs(SAVE_DIR, exist_ok=True)


# 融合函数
def fuse_features(deep_path, trad_path, save_path, split_name):
    print(f"\nProcessing {split_name} set...")

    deep_data = torch.load(deep_path)
    trad_data = torch.load(trad_path)

    # ---------- 取特征 ----------
    # 深度特征
    deep_feats = deep_data["features"]        # [N, D1]
    deep_labels = deep_data["labels"]

    # 传统特征（只用 SIFT）
    if "sift" in trad_data:
        trad_feats = trad_data["sift"]    # [N, D2]
    else:
        raise KeyError("Traditional feature file must contain 'sift' key")

    trad_labels = trad_data["labels"]

    # ---------- 一致性检查 ----------
    assert len(deep_feats) == len(trad_feats), \
        f"Feature number mismatch: {len(deep_feats)} vs {len(trad_feats)}"

    if not torch.equal(deep_labels, trad_labels):
        diff = (deep_labels != trad_labels).nonzero(as_tuple=True)[0][:10]
        raise ValueError(
            f"Label mismatch detected in {split_name} set, "
            f"first mismatch indices: {diff.tolist()}"
        )

    print(f"✔ {split_name}: feature & label alignment verified")

    # ---------- 特征归一 ----------
    deep_feats = torch.nn.functional.normalize(deep_feats, dim=1)
    trad_feats = torch.nn.functional.normalize(trad_feats, dim=1)

    # ---------- 特征拼接 ----------
    fused_feats = torch.cat([deep_feats, trad_feats], dim=1)

    print(
        f"{split_name} fused feature shape: {fused_feats.shape} "
        f"(deep={deep_feats.shape[1]}, sift={trad_feats.shape[1]})"
    )

    # ---------- 保存 ----------
    torch.save(
        {
            "feat": fused_feats,
            "labels": deep_labels
        },
        save_path
    )

    print(f"{split_name} fused features saved to: {save_path}")



if __name__ == "__main__":
    fuse_features(
        TRAIN_DEEP_PATH,
        TRAIN_TRAD_PATH,
        os.path.join(SAVE_DIR, "train_feature_fusion.pt"),
        split_name="train"
    )

    fuse_features(
        TEST_DEEP_PATH,
        TEST_TRAD_PATH,
        os.path.join(SAVE_DIR, "test_feature_fusion.pt"),
        split_name="test"
    )

    print("\n[Done] Feature-level fusion completed successfully.")






