from .dataset import Dataset
import torch
import torchvision.transforms as transforms


def generate_loader(opt):
    dataset = Dataset
    img_size = opt.input_size

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    dataset = dataset(opt, transform=transform)

    kwargs = {
        "batch_size": opt.batch_size,
        "shuffle": True,
        "drop_last": True,
    }

    return torch.utils.data.DataLoader(dataset, **kwargs)
