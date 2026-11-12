# 任务卡：把 vae.py 第65–77行、第89–101行的两段"看图结构"搬进来
# 只管一件事：进来的是图片数字，出去的也是图片数字；别的都不管
# 照抄自：brain-net-diffusion 的 src/models/components/
import torch
import torch.nn as nn


class Encoder(nn.Module):
    
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, ch, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(ch, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.Conv2d(ch *2, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.Conv2d(ch * 4, ch * 8, 4, 2, 1),
            nn.BatchNorm2d(ch * 8),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    
    def __init__(self, ch=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(ch * 8, ch * 4, 4, 2, 1),
            nn.BatchNorm2d(ch * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, 2, 1),
            nn.BatchNorm2d(ch * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch * 2, ch, 4, 2, 1),
            nn.BatchNorm2d(ch),
            nn.ReLU(True),
            nn.ConvTranspose2d(ch, 3, 4, 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    fake_image = torch.zeros(1,3,64,64)
    enc = Encoder(ch=64)    #实例化
    feat = enc(fake_image)
    print('Encoder output shape:', feat.shape)
    dec = Decoder(ch=64)
    img = dec(feat)
    print('Decoder output shape:', img.shape)

