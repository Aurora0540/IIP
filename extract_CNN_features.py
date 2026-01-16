from dataloader import ParquetImageDataset
import torch
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from CNN_feature import resnet50_cifar100_feature,FeatureExtractor

train_set = ParquetImageDataset(
    "C:/Users/18320/Desktop/IIP/CIFAR100/train-00000-of-00001.parquet",
    train=False)
test_set = ParquetImageDataset(
    "C:/Users/18320/Desktop/IIP/CIFAR100/test-00000-of-00001.parquet",
    train=False)

def extract_CNN_features(model, save_path):
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)
    feats = []
    labels = []

    model.eval()
    with torch.no_grad():
        for imgs, lbls in tqdm(test_loader, desc="Extracting train features"):
            imgs = imgs.cuda()
            f = model(imgs)
            feats.append(f.cpu())
            labels.append(lbls)

    feats = torch.cat(feats)
    labels = torch.cat(labels)

    torch.save({
        "features": feats,
        "labels": labels
    }, save_path)

    print(f"Saved train features to {save_path}")



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = resnet50_cifar100_feature().to(device)
    ckpt = torch.load("C:/Users/18320/Desktop/IIP/resnet50_cifar100_layer4_fc.pth", map_location=device)
    model.load_state_dict(ckpt, strict=False)

    extractor = FeatureExtractor(model).to(device)


    extract_CNN_features(extractor,save_path='CNN_test_features.pt')
    #extract_CNN_features(extractor,save_path='CNN_test_features.pt')