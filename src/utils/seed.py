#新增：固定随机性
#（照抄范本一 ae.py 第29–36行）
import random

import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    set_seed(42)
    a = torch.rand(3)
    set_seed(42)
    b = torch.rand(3)
    print("第一次：", a)
    print("第二次；",b)
    print("两次是否相等：", torch.equal(a, b))