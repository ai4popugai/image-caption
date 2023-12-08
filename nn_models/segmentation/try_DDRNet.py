import torch

from ddrnet.models import DDRNet23


def main():
    model = DDRNet23(num_classes=19)
    t = torch.rand(1, 3, 256, 256)
    model.eval()
    t_out = model(t)
    print(t_out.shape)


if __name__ == '__main__':
    main()
