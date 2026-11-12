#认领第137–148行
#存档、拼图保存
import torch
from torchvision import utils

def save_checkpoint(model, optim, epoch, path):

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optim_state": optim.state_dict()
    }

    torch.save(state, path)


def save_image_grid(tensor, filename, nrow=8):

    tensor = torch.clamp(tensor, 0, 1)  # clip范围（0，1），防鬼图
    utils.save_image(tensor, filename, nrow=nrow, padding=2)


if __name__ == "__main__":

    model = torch.nn.Linear(4, 2)
    optim = torch.optim.SGD(model.parameters(), lr=0.1)
    save_checkpoint(model, optim, epoch=3, path="/tmp/mini_ckpt.pth")
    loaded = torch.load("/tmp/mini_ckpt.pth")
    print("存档里有什么：", list(loaded.keys()))
    print("存到第几轮:", loaded["epoch"])
