#认领第11–24、152–211行
#读清单 → 领零件 → 训练循环 → 存档出图 + 启动入口
#略
import argparse
import os
from datetime import datetime

import torch
import yaml

from src.data.celeba import build_dataloader
from src.models.vae import ConvVAE
from src.utils.checkpoint import save_checkpoint, save_image_grid
from src.utils.losses import vae_loss
from src.utils.seed import set_seed

def load_yaml(path):    #造一个读配置的零件，因为配置文件是文本，但python要吃字典，所以需要转换一下
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_argparser():  #在命令行暴露三个入口，可以按需传入不同的数据、模型、训练流程
    parser = argparse.ArgumentParser(description="训练部件化 VAE")
    parser.add_argument("--data", default="configs/data/celeba.yaml")
    parser.add_argument("--train",default="configs/train/default.yaml")
    parser.add_argument("--model", default="configs/model/vae.yaml")
    return parser


def main(args):
    cfg = {}    #接收传入的参数，并用load_yaml转化为字典
    cfg.update(load_yaml(args.data))
    cfg.update(load_yaml(args.model))
    cfg.update(load_yaml(args.train))

    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["save_dir"], exist_ok=True)


    print("训练前清点一下configs：")
    for k, v in cfg.items():
        print(f" {k}: {v}")


    loader = build_dataloader(cfg)
    model = ConvVAE(
        latent_dim=cfg["latent_dim"],
        ch=cfg["ch"],
        image_size=cfg["image_size"],
    ).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    global_step = 0
    for epoch in range(1, cfg["num_epochs"] + 1):
        model.train()   #model继承nn.module，自带.train()方法
        epoch_loss, epoch_recon, epoch_kld = 0.0, 0.0, 0.0  #每轮开始前先进行初始化

        for batch_idx, imgs in enumerate(loader):   #imgs是一个图像队列，一个batch_size有1024张
            imgs = imgs.to(device, non_blocking=True)
            optim.zero_grad()   #把上一batch回传攒下的梯度清零，防止跨batch累加
            recon, mu, logvar = model(imgs) #过模型
            loss, recon_l, kld = vae_loss(recon, imgs, mu, logvar)  #算loss
            loss.backward() #以loss为线索倒推，算出每个参数该往哪调多少
            optim.step()    #优化器更新参数

            epoch_loss += loss.item()   #把当前batch的损失记到当前epoch的总损失上
            epoch_recon += recon_l.item()
            epoch_kld += kld.item()
            global_step += 1    #跑完一个batch记一步

            if batch_idx % 100 == 0:    #每跑100步显示一次进度
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"Epoch {epoch}/{cfg['num_epochs']} | Batch {batch_idx}/{len(loader)} | "
                    f"Loss {loss.item():.4f} | "
                    f"(recon {recon_l.item():.4f}, kld {kld.item():.4f})"
                )
    

        #一个epoch跑完后
        n = len(loader.dataset)
        print(
            f"=== Epoch {epoch} 完成。平均扣分:{epoch_loss/n:.4f}"
            f"(recon {epoch_recon/n:.4f}, kld {epoch_kld/n:.4f}) ==="
        )
        #检查当前epoch是否到达存档点？
        if epoch % cfg["sample_every"] == 0 or epoch == 1:  #第一个epoch必存档，及早发现问题防止后面跑偏
            ckpt_path = os.path.join(cfg["save_dir"], f"vae_epoch{epoch}.pth")
            save_checkpoint(model, optim, epoch, ckpt_path)

            #在存档点验证一下，记录结果
            #自带继承nn.module，切换到验证模式
            model.eval()
            with torch.no_grad():   #推理时关闭
                ###测试一：重建测试（有对比图）
                imgs = next(iter(loader)).to(device)[:cfg["num_sample_images"]] #从loader里取出shuffle打乱后第一个batch的前num_sample_images图，和模型重建图作对比
                recon, _, _ = model(imgs)   #只收重建图，返回的另外两个结果不收，占位即可
                combined = torch.cat([imgs, recon], dim=0)
                save_image_grid(combined, os.path.join(cfg["save_dir"], f"recon_epoch{epoch}.png"))
                
                ###测试二：生成测试（无对比图）
                z = torch.randn(cfg["num_sample_images"], cfg["latent_dim"], device=device)
                samples = model.decode(z)   #只用解码器从隐空间生成即可
                save_image_grid(samples, os.path.join(cfg["save_dir"], f"sample_epoch{epoch}.png"))
            model.train()   #验证完切换回训练模式
        

    print("训练完成。")


if __name__ == "__main__":
    main(build_argparser().parse_args())
