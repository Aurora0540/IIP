import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from dataloader import ParquetImageDataset
from torchvision.models import resnet50

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# nohup python train.py > 2.log 2>&1 &

# 主训练逻辑
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Dataset
    train_set = ParquetImageDataset(
        "/home/krli/IIP/CIFAR100/train-00000-of-00001.parquet",
        train=True
    )
    test_set = ParquetImageDataset(
        "/home/krli/IIP/CIFAR100/test-00000-of-00001.parquet",
        train=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_set,
        batch_size=1000,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )


    # Model
    def resnet50_cifar100():
        model = resnet50(weights=None)

        # 修改 conv1
        model.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )

        # 删除 maxpool
        model.maxpool = nn.Identity()

        # 修改 fc
        model.fc = nn.Linear(model.fc.in_features, 100)

        return model

    model = resnet50_cifar100()

    # 加载 ImageNet 预训练权重
    ckpt = torch.load("/home/krli/IIP/ResNet50/pytorch_model.bin", map_location="cpu")

    # 兼容 state_dict / 包一层的情况
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    # 删除fc
    ckpt = {
        k: v for k, v in ckpt.items()
        if not (k.startswith("conv1.") or k.startswith("fc."))
    }

    msg = model.load_state_dict(ckpt, strict=False)
    print("Load pretrained:", msg)

    model.to(device)


    # 冻结参数
    for name, param in model.named_parameters():
        if not name.startswith(("conv1.", "layer3", "layer4","fc")):
            param.requires_grad = False

    params_to_update = model.parameters()


    # Optimizer
    optimizer = optim.SGD(
        params_to_update,
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss()


    # Train
    epochs = 1000
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=0.00001)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        epoch_time = time.time() - epoch_start_time

        print(
            f"[Epoch {epoch+1}/{epochs}] "
            f"Loss: {running_loss / len(train_loader):.4f}"
            f" Time: {epoch_time:.1f} s"
        )


        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                preds = outputs.argmax(dim=1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()

        acc = 100.0 * correct / total
        print(f"Test Acc: {acc:.2f}%")

    torch.save(model.state_dict(), "checkpoints/resnet50_cifar100_class100_1000.pth")
    print("Training finished.")


if __name__ == "__main__":
    main()
