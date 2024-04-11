import torch
from torch.nn import functional as F


if __name__ == "__main__":

    inp = torch.randn((1, 3, 256, 256))
    
    # net(inp)

    patches = F.unfold(
            inp,
            kernel_size=2,
            stride=2,
        )

    print(patches.shape)

