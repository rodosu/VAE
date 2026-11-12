#认领第129–133行
#算扣分（还原差多少 + 一项正则）
import torch
import torch.nn.functional as F


def vae_loss(recon_x, x, mu, logvar):

    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kld, recon_loss, kld

if __name__ == "__main__":
    
    fake_recon = torch.ones(1, 3, 64, 64)
    fake_x = torch.zeros(1, 3, 64, 64)
    fake_mu = torch.zeros(1, 20)
    fake_logvar = torch.zeros(1, 20)
    total, recon_loss, kld = vae_loss(fake_recon, fake_x, fake_mu, fake_logvar)
    print("recon_loss:", recon_loss.item())
    print("kld:", kld.item())
    print("total loss:", total.item())