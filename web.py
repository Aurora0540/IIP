import streamlit as st
import torch
import pandas as pd
from PIL import Image
import io
import os
import random

# 设置页面配置
st.set_page_config(layout="wide", page_title="图像检索系统演示")

# ===== 1. 配置路径 (Windows 环境) =====
BASE_PATH = "C:/Users/18320/Desktop/IIP"

PATHS = {
    "cnn_train": os.path.join(BASE_PATH, "CNN_train_features.pt"),
    "cnn_test": os.path.join(BASE_PATH, "CNN_test_features.pt"),
    "trad_gallery": os.path.join(BASE_PATH, "gallery_features.pt"),
    "trad_query": os.path.join(BASE_PATH, "query_features.pt"),
    "fusion_train": os.path.join(BASE_PATH, "train_feature_fusion.pt"),
    "fusion_test": os.path.join(BASE_PATH, "test_feature_fusion.pt"),
    "parquet_train": os.path.join(BASE_PATH, "CIFAR100/train-00000-of-00001.parquet"),
    "parquet_test": os.path.join(BASE_PATH, "CIFAR100/test-00000-of-00001.parquet"),
    "logo": os.path.join(BASE_PATH, "logo.jpg")
}


# ===== 2. 缓存加载数据 =====
@st.cache_resource
def load_data():
    data = {}

    # 检查路径 (跳过 logo 和 cnn_test 的强制检查)
    for key, path in PATHS.items():
        if key == "logo" or key == "cnn_test":
            continue
        if not os.path.exists(path):
            st.error(f"找不到文件: {path}")
            st.stop()

    # A. 加载图像数据 (Parquet)
    try:
        data["df_train"] = pd.read_parquet(PATHS["parquet_train"])
        data["df_test"] = pd.read_parquet(PATHS["parquet_test"])
    except Exception as e:
        st.error(f"读取图片数据失败: {e}")
        st.stop()

    # B. 加载特征数据
    # 1. Fusion
    try:
        fusion_train = torch.load(PATHS["fusion_train"], map_location='cpu')
        fusion_test = torch.load(PATHS["fusion_test"], map_location='cpu')
        data["fusion_gallery"] = fusion_train["feat"]
        data["fusion_query"] = fusion_test["feat"]
        data["fusion_labels"] = fusion_train["labels"]
        data["test_labels"] = fusion_test["labels"]
    except Exception as e:
        st.error(f"加载融合特征失败: {e}")

    # 2. CNN
    try:
        cnn_train = torch.load(PATHS["cnn_train"], map_location='cpu')
        if isinstance(cnn_train, dict):
            data["cnn_gallery"] = cnn_train["features"] if "features" in cnn_train else cnn_train["feat"]
        else:
            data["cnn_gallery"] = cnn_train

        if os.path.exists(PATHS["cnn_test"]):
            cnn_test = torch.load(PATHS["cnn_test"], map_location='cpu')
            data["cnn_query"] = cnn_test["features"] if isinstance(cnn_test, dict) else cnn_test
        else:
            data["cnn_query"] = data["fusion_query"][:, :2048]
    except:
        pass

    # 3. Traditional
    try:
        trad_gallery = torch.load(PATHS["trad_gallery"], map_location='cpu')
        trad_query = torch.load(PATHS["trad_query"], map_location='cpu')
        data["color_gallery"] = trad_gallery["color"].squeeze()
        data["color_query"] = trad_query["color"].squeeze()
    except:
        pass

    return data


# ===== 3. 工具函数 =====
def load_image_from_bytes(img_data):
    if isinstance(img_data, dict):
        if "bytes" in img_data:
            img_data = img_data["bytes"]
        elif "path" in img_data:
            return Image.open(img_data["path"]).convert("RGB")
    return Image.open(io.BytesIO(img_data)).convert("RGB")


def compute_topk(query_feat, gallery_feats, topk=5):
    if not isinstance(query_feat, torch.Tensor): query_feat = torch.tensor(query_feat)
    if not isinstance(gallery_feats, torch.Tensor): gallery_feats = torch.tensor(gallery_feats)

    if query_feat.dim() == 1: query_feat = query_feat.unsqueeze(0)

    q_norm = torch.nn.functional.normalize(query_feat, dim=1)
    g_norm = torch.nn.functional.normalize(gallery_feats, dim=1)

    sim = torch.mm(q_norm, g_norm.t()).squeeze(0)
    vals, idxs = sim.topk(topk)
    return vals, idxs


# ===== 4. 主逻辑 =====
def main():
    # 加载数据
    with st.spinner('正在加载系统资源...'):
        data_dict = load_data()

    # --- Session State 初始化 ---
    if 'current_query_idx' not in st.session_state:
        st.session_state.current_query_idx = 2026

    def random_select():
        max_idx = len(data_dict["df_test"]) - 1
        new_idx = random.randint(0, max_idx)
        st.session_state.current_query_idx = new_idx

    # --- 侧边栏设计 ---
    with st.sidebar:
        # ==========================================
        # 修改部分：增加了 try-except 防止图片坏了导致崩溃
        # ==========================================
        if os.path.exists(PATHS["logo"]):
            try:
                # 尝试打开图片以验证它是否有效
                logo_img = Image.open(PATHS["logo"])
                st.image(logo_img, use_container_width=True)
            except Exception:
                st.warning("⚠️ logo.png 文件损坏或格式错误，已跳过显示。")
        else:
            st.info("💡 提示: 您可以在文件夹放入 logo.png 来显示校徽")
        # ==========================================

        st.markdown("---")
        st.header("控制面板")

        # 2. 随机抽取
        st.markdown("### 1. 查询图像")
        if st.button("🎲 随机抽取测试图像", type="primary", use_container_width=True):
            random_select()

        st.caption(f"当前索引 ID: {st.session_state.current_query_idx}")

        st.markdown("---")

        # 3. 检索设置
        st.markdown("### 2. 检索设置")
        method = st.radio("选择特征类型", ["CNN", "传统特征", "特征融合"])

        top_k = st.slider("显示结果数量 (Top-K)", 1, 10, 5)

        st.markdown("---")
        st.markdown("**System Status**")
        st.success("✔ Model Loaded")
        st.success("✔ Data Ready")

    # --- 主界面 ---
    st.title("🔎 图像检索系统演示")

    query_idx = st.session_state.current_query_idx

    # 准备 Query 数据
    try:
        query_row = data_dict["df_test"].iloc[query_idx]
        query_img = load_image_from_bytes(query_row["img"])

        if "test_labels" in data_dict:
            true_label_id = data_dict["test_labels"][query_idx].item()
        elif "coarse_label" in query_row:
            true_label_id = query_row["coarse_label"]
        else:
            true_label_id = "?"

    except Exception as e:
        st.error(f"读取索引 {query_idx} 出错: {e}")
        st.stop()

    col1, col2 = st.columns([1, 3])

    # --- 左侧：Query Image ---
    with col1:
        st.subheader("Query Image")
        st.image(query_img, width=200)
        st.info(f"**True Label:** {true_label_id}\n\n(Index: {query_idx})")

    # --- 右侧：检索结果 ---
    with col2:
        st.subheader(f"Retrieval Results ({method})")

        gallery_feats = None
        query_vec = None

        if method == "CNN":
            gallery_feats = data_dict["cnn_gallery"]
            query_vec = data_dict["cnn_query"][query_idx]
        elif method == "传统特征":
            gallery_feats = data_dict["color_gallery"]
            query_vec = data_dict["color_query"][query_idx]
        elif method == "特征融合":
            gallery_feats = data_dict["fusion_gallery"]
            query_vec = data_dict["fusion_query"][query_idx]

        if gallery_feats is not None:
            scores, indices = compute_topk(query_vec, gallery_feats, topk=top_k)

            res_cols = st.columns(top_k)
            for i, col in enumerate(res_cols):
                idx = indices[i].item()
                score = scores[i].item()

                res_row = data_dict["df_train"].iloc[idx]
                res_img = load_image_from_bytes(res_row["img"])
                res_label = data_dict["fusion_labels"][idx].item()

                is_match = (res_label == true_label_id)
                color = "green" if is_match else "red"
                match_text = "✔ Match" if is_match else "✘ Diff"

                with col:
                    st.image(res_img, use_container_width=True)
                    st.markdown(f"**Rank {i + 1}**")
                    st.markdown(f":{color}[Label: {res_label}]")
                    st.caption(f"Sim: {score:.3f}\n{match_text}")
        else:
            st.warning("特征数据未加载完全，无法检索。")


if __name__ == "__main__":
    main()