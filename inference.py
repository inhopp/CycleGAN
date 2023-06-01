import os
import torch
from option import get_option
from model import Generator
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms


@torch.no_grad()
def main(opt):
    dev = torch.device("cuda:{}".format(opt.gpu)
                       if torch.cuda.is_available() else "cpu")
    ft_path = os.path.join(opt.ckpt_root, "Gen_A.pt")

    model = Generator(in_channels=3, features=64).to(dev)

    model.load_state_dict(torch.load(ft_path))

    # data
    img_path = os.path.join("./temp.jpeg")
    image = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize(opt.input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ])

    image = transform(image).to(dev)

    model.eval()

    output = model(image)
    torchvision.utils.save_image(output, f"output.png")

    print("########## inference Finished ###########")


if __name__ == '__main__':
    opt = get_option()
    main(opt)
