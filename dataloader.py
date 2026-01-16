import torchvision.transforms as transforms
import io
import base64
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pyarrow.parquet as pq


class ParquetImageDataset(Dataset):
    def __init__(self, parquet_path, train=True):
        # ===== 1. 只在主进程读 parquet（关键） =====
        table = pq.read_table(parquet_path)
        data = table.to_pydict()


        self.images = data["img"]
        self.labels = data["coarse_label"]

        assert len(self.images) == len(self.labels)

        # ===== 2. transforms =====
        if train:
            self.transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2673, 0.2564, 0.2761]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.5071, 0.4865, 0.4409],
                    std=[0.2673, 0.2564, 0.2761]
                )
            ])

    def __len__(self):
        return len(self.labels)

    def _decode_image(self, img):
        if isinstance(img, dict):
            if "bytes" in img and img["bytes"] is not None:
                img = img["bytes"]
            elif "path" in img and img["path"] is not None:
                img = img["path"]
            else:
                raise ValueError(f"Invalid image dict keys: {img.keys()}")

        # case 1: base64 / path 字符串
        if isinstance(img, str):
            # 文件路径
            if img.endswith((".png", ".jpg", ".jpeg")):
                img = Image.open(img).convert("RGB")
            else:
                # base64
                img_bytes = base64.b64decode(img)
                img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # case 2: bytes
        elif isinstance(img, (bytes, bytearray)):
            img = Image.open(io.BytesIO(img)).convert("RGB")

        # case 3: numpy array
        elif isinstance(img, np.ndarray):
            img = Image.fromarray(img).convert("RGB")

        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

        return img

    def __getitem__(self, idx):
        img = self.images[idx]
        label = int(self.labels[idx])

        img = self._decode_image(img)
        img = self.transform(img)

        return img, label

