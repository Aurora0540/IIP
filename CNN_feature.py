import random
import torch
from torch import nn
from torchvision.models import resnet50
from dataloader import ParquetImageDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

train_set = ParquetImageDataset(
    "C:/Users/18320/Desktop/IIP/CIFAR100/train-00000-of-00001.parquet",
    train=False)
test_set = ParquetImageDataset(
    "C:/Users/18320/Desktop/IIP/CIFAR100/test-00000-of-00001.parquet",
    train=False)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def denormalize(img):
    CIFAR100_MEAN = [0.5071, 0.4865, 0.4409]
    CIFAR100_STD = [0.2673, 0.2564, 0.2761]
    img = img.clone()
    for c in range(3):
        img[c] = img[c] * CIFAR100_STD[c] + CIFAR100_MEAN[c]
    return img.clamp(0, 1)

def resnet50_cifar100_feature():
    model = resnet50(weights=None)

    model.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 20)

    return model


class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.backbone = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool
        )

    def forward(self, x):
        x = self.backbone(x)
        x = torch.flatten(x, 1)  # [B, 2048]
        x = F.normalize(x, dim=1)  # 归一化处理
        return x

def show_by_index(test_set, train_set, query_idx, topk_idxs, title):
    query_img, query_label = test_set[query_idx]

    top_imgs = []
    top_labels = []

    for idx in topk_idxs:
        img, lbl = train_set[idx]
        top_imgs.append(img)
        top_labels.append(lbl)

    n = len(top_imgs) + 1
    plt.figure(figsize=(15, 3))

    # Query
    plt.subplot(1, n, 1)
    plt.imshow(denormalize(query_img).permute(1, 2, 0))
    plt.title(f"Query\nLabel: {query_label}")
    plt.axis("off")

    # Top-k
    for i, (img, lbl) in enumerate(zip(top_imgs, top_labels)):
        plt.subplot(1, n, i + 2)
        plt.imshow(denormalize(img).permute(1, 2, 0))
        plt.title(f"Top-{i+1}\nLabel: {lbl}")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()



def CNN_retrieve_topk(model, feature_file="CNN_train_features.pt", num_query=10, topk=5, metric='cosine',seed=2026):
    data = torch.load(feature_file)
    train_feats = data["features"]   # [50000, 2048]
    train_labels = data["labels"]

    set_seed(seed)
    indices = random.sample(range(len(test_set)), num_query)

    model.eval()
    with torch.no_grad():
        print('-' * 60)
        print("CNN深度特征检索:")
        print('-' * 60)
        for qi in indices:
            img, label = test_set[qi]
            img = img.unsqueeze(0).cuda()

            q_feat = model(img)   # [1, 2048]

            # 余弦相似度
            if metric == "cosine":
                sim = torch.mm(q_feat.cpu(), train_feats.t()).squeeze(0)
                vals, idxs = sim.topk(topk)

            # 欧氏距离
            if metric == "euclidean":
                dist = torch.cdist(q_feat.cpu(), train_feats).squeeze(0)  # [1, N]
                vals, idxs = dist.topk(topk, largest=False)

            show_by_index(test_set,
                          train_set,
                          qi, idxs,
                          title=f"CNN + {metric.upper()} | Query idx={qi}")
            print(f"\nQuery label: {label}")
            print("Top-5 matches:")
            for i in range(topk):
                print(
                    f"  Rank {i+1}: "
                    f"Index={idxs[i].item()}, "
                    f"Label={train_labels[idxs[i]].item()}, "
                    f"Similarity or distance={vals[i].item():.4f}"
                )

if __name__ == '__main__':
    # CNN深度特征检索
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50_cifar100_feature().to(device)
    ckpt = torch.load("C:/Users/18320/Desktop/IIP/resnet50_cifar100_layer4_fc.pth", map_location=device)
    model.load_state_dict(ckpt, strict=False)
    extractor = FeatureExtractor(model).to(device)
    CNN_retrieve_topk(extractor, num_query=5, topk=10, metric="cosine", seed=2026)