#认领第26–53行
#裁图规矩 + 数据集 + 递图助手
#略

import glob
import os

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

class CelebADataset(Dataset):

    def __init__(self, root, transform=None):
        self.paths = sorted(glob.glob(os.path.join(root, '*.jpg')))
        if len(self.paths) == 0:
            raise FileNotFoundError(f"在{root}中找不到任何图像文件。请检查路径是否正确。")
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img  = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img



def build_transform(image_size=64):
    ##裁剪格式
    return transforms.Compose([
        transforms.Resize(image_size),  #按短边缩放（178×218 → 64×78）
        transforms.CenterCrop(image_size),  #中间裁方（64×78 → 64×64）
        transforms.ToTensor(),
    ])


def build_dataloader(cfg):
    dataset = CelebADataset(
        root = cfg["data_root"],
        transform = build_transform(cfg["image_size"])
    )

    loader = DataLoader(
        dataset,
        batch_size = cfg["batch_size"],
        shuffle = True,
        num_workers = cfg.get("num_workers", 8),
        pin_memory = True
    )

    return loader

if __name__ == "__main__":
    cfg = {
        "data_root": "/home/rodosu/projects/VAE/faces/img_align_celeba",
        "image_size": 64,
        "batch_size": 8 
    }
    loader = build_dataloader(cfg)
    batch  = next(iter(loader))
    print("一批的形状：", batch.shape)
    print("数值范围：", batch.min().item(), "~", batch.max().item())
