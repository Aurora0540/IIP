import torch
import torchvision.transforms as T
import random
import matplotlib.pyplot as plt
import os
from dataloader import ParquetImageDataset
from tqdm import tqdm


SEED = 100
random.seed(SEED)
torch.manual_seed(SEED)

SAVE_DIR = "C:/Users/18320/Desktop/IIP/CIFAR100"
os.makedirs(SAVE_DIR, exist_ok=True)

TRAIN_BAK_PATH = os.path.join(SAVE_DIR, "train-00000-of-00001.parquet.bak")
TEST_BAK_PATH  = os.path.join(SAVE_DIR, "test-00000-of-00001.parquet.bak")
train_path = "C:/Users/18320/Desktop/IIP/CIFAR100/train-00000-of-00001.parquet"
test_path = "C:/Users/18320/Desktop/IIP/CIFAR100/test-00000-of-00001.parquet"
train_set = ParquetImageDataset(
    "C:/Users/18320/Desktop/IIP/CIFAR100/train-00000-of-00001.parquet",
    train=False)
test_set = ParquetImageDataset(
    "C:/Users/18320/Desktop/IIP/CIFAR100/test-00000-of-00001.parquet",
    train=False)

def denormalize(tensor, mean, std):
    mean = torch.tensor(mean).view(3, 1, 1)
    std = torch.tensor(std).view(3, 1, 1)
    return tensor * std + mean

# 高斯模糊
class AddGaussianNoise(object):
    def __init__(self, mean=0., std=0.05):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.std + self.mean


augment_transform = T.Compose([
    T.RandomRotation(20),
    T.RandomHorizontalFlip(p=0.5),
    T.RandomResizedCrop(32, scale=(0.8, 1.0)),
    AddGaussianNoise(0., 0.05),
    T.Lambda(lambda x: torch.clamp(x, 0.0, 1.0))
])


# 预处理并保存为 .bak
def process_and_save(dataset, save_path, split_name):
    print(f"Processing {split_name} set...")

    original_imgs = []
    augmented_imgs = []
    labels = []


    for img, label in tqdm(dataset):
        original_imgs.append(img)
        img = augment_transform(img)
        augmented_imgs.append(img)
        labels.append(label)

    data = {
        "original": torch.stack(original_imgs),
        "augmented": torch.stack(augmented_imgs),
        "labels": torch.tensor(labels)
    }

    torch.save(data, save_path)
    print(f"✔ {split_name} saved to {save_path}")

if not os.path.exists('C:/Users/18320/Desktop/IIP/CIFAR100/train-00000-of-00001.parquet.bak'):
    process_and_save(train_set, TRAIN_BAK_PATH, "train")
    process_and_save(test_set, TEST_BAK_PATH, "test")


# 随机展示 5 组 原图 vs 增强图
def show_samples(dataset,bak_path, title):
    data = torch.load(bak_path)
    idxs = random.sample(range(len(data["labels"])), 5)

    plt.figure(figsize=(10, 4))
    for i, idx in enumerate(idxs):
        # 原图
        img,_ = dataset[idx]
        img = denormalize(
            img,
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.permute(1, 2, 0))
        plt.title("Original")
        plt.axis("off")

        # 增强图
        plt.subplot(2, 5, i + 6)
        plt.imshow(data["augmented"][idx].permute(1, 2, 0))
        plt.title("Augmented")
        plt.axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


show_samples(train_set,TRAIN_BAK_PATH, "Train Set Augmentation Examples")
show_samples(test_set,TEST_BAK_PATH, "Test Set Augmentation Examples")
