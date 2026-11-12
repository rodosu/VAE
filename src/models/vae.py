#认领第57–64、79–86、103–125行
#把零件拼成完整 VAE
#略

import torch
import torch.nn as nn

from src.models.components.conv_blocks import Encoder, Decoder

class ConvVAE(nn.Module):
    ###"图片 → 压缩 → 取基因 → 还原"###
    
    def __init__(self, latent_dim=128, ch=64, image_size=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.ch = ch
        self.image_size = image_size

        self.encoder = Encoder(ch=ch)
        self.decoder = Decoder(ch=ch)

        ### 让代码自己数：假图过一遍压缩零件，看输出几格（512×4×4 = 8192）
        with torch.no_grad():   #关掉梯度计算，节省显存
            dummy = torch.zeros(1, 3, image_size, image_size)   #造一个假图，1张3通道64×64的黑图
            feat = self.encoder(dummy)  #（1, 512, 4, 4）   #经过encoder压缩，输出特征图
            self._feat_shape = feat.shape[1:]   #切片(1, 512, 4, 4) → (512, 4, 4)，记encoder输出的特征图形状，后面解码器要用
            feat_dim = feat.view(1, -1).size(1)  #‘-1’自动计算这一维的长度(1, 512, 4, 4) → (1, 8192)，size(1)取第二维的长度8192

        self.fc_mu = nn.Linear(feat_dim, latent_dim)   #浓缩块 → 雾的中心
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)   #浓缩块 → 雾的胖瘦
        self.fc_dec = nn.Linear(latent_dim, feat_dim)   #基因 → 浓缩块

    def encode(self, x):
        h = self.encoder(x) #(x, 512, 4, 4)
        h = h.view(h.size(0), -1)   #展平成 (x, 8192)
        mu = self.fc_mu(h)   #(x, latent_dim)
        logvar = self.fc_logvar(h)   #(x, latent_dim)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)   #重参数化引入外部随机性
        return mu + eps * std   

    def decode(self, z):
        h = self.fc_dec(z)
        h = h.view(h.size(0),*self._feat_shape) #从(1, 8192) → (1, 512, 4, 4)
        x_recon = self.decoder(h)
        return x_recon
    
    def forward(self, x):   #把图片数字进来，经过编码器压缩、取基因、解码器还原，出去的还是图片数字
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


if __name__ == "__main__":
    model = ConvVAE(latent_dim=128, ch=64, image_size=64)
    fake = torch.zeros(1, 3, 64, 64)

    recon, mu, logvar = model(fake)
    print('Reconstructed image shape:', recon.shape)
    print('Latent mean shape:', mu.shape)
    print('Latent logvar shape:', logvar.shape)

    z = torch.randn(1, 128)
    img = model.decode(z)
    print('Decoded image shape from random latent vector:', img.shape)

