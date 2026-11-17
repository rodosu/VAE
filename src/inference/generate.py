import argparse

import torch
import yaml
from PIL import Image

from src.data.celeba import build_transform   
from src.models.vae import ConvVAE
from src.utils.checkpoint import save_image_grid

def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_model(model_cfg, ckpt_path, device):
    model = ConvVAE(
        latent_dim=model_cfg["latent_dim"],
        ch=model_cfg["ch"],
        image_size=model_cfg["image_size"]
    ).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"已加载：{ckpt_path} (训练到第{ckpt['epoch']}轮)")
    return model

@torch.no_grad()
def generate_faces(model, n, cfg, device):
    z = torch.randn(n,cfg["latent_dim"],device=device)
    return model.decode(z)

@torch.no_grad()
def reconstruct(model, img_path, cfg, device):
    img = Image.open(img_path).convert("RGB")
    x = build_transform(cfg["image_size"])(img)
    x = x.unsqueeze(0).to(device)   # (3,64,64)→(1,3,64,64)
    recon, mu, logvar = model(x)
    return x, recon


def main():
    parser = argparse.ArgumentParser(description="VAE 推理：加载ckpt，可执行 重建/生成 模式")
    parser.add_argument("--model", default="configs/model/vae.yaml", help="模型配置")
    parser.add_argument("--ckpt", default="vae_test_checkpoints/vae_epoch300.pth",help="模型位置")
    parser.add_argument("--n", type=int, default=8, help="生成几张")
    parser.add_argument("--data", default="configs/data/celeba.yaml", help="生成图片的大小")
    parser.add_argument("--recon", default=None, help="用来评估重建质量的图片")
    parser.add_argument("--out", default=None, help="输出文件名")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = load_yaml(args.model)
    cfg.update(load_yaml(args.data))
    model = load_model(cfg, args.ckpt, device)  #传入模型配置，模型权重，并在加载到GPU上

    if args.recon:  #重建模式
        x, recon = reconstruct(model, args.recon, cfg, device)
        out = args.out or "recon_result.png"
        save_image_grid(torch.cat([x, recon]), out, nrow=2)
        print(f"重建对比已保存至:{out}")
    else:           #生成模式（默认）
        faces = generate_faces(model, args.n, cfg, device)
        out = args.out or f"faces_{args.n}.png"
        save_image_grid(faces, out)
        print(f"已生成 {args.n} 张新图：{out}")


if __name__ == "__main__":
    main()

    


