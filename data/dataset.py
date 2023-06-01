import os
from PIL import Image
import torch.utils.data as data


class Dataset(data.Dataset):
    def __init__(self, opt, transform):
        self.data_dir_A = opt.data_dir + '/trainA'
        self.list_files_A = os.listdir(self.data_dir_A)

        self.data_dir_B = opt.data_dir + '/trainB'
        self.list_files_B = os.listdir(self.data_dir_B)

        self.A_len = len(self.list_files_A)
        self.B_len = len(self.list_files_B)
        self.length = max(self.A_len, self.B_len)

        self.transfrom = transform

    def __getitem__(self, index):
        A_img_file = self.list_files_A[index % self.A_len]
        B_img_file = self.list_files_B[index & self.B_len]

        A_img_path = os.path.join(self.data_dir_A, A_img_file)
        B_img_path = os.path.join(self.data_dir_B, B_img_file)

        imageA = Image.open(A_img_path).convert('RGB')
        imageB = Image.open(B_img_path).convert('RGB')

        imageA = self.transfrom(imageA)
        imageB = self.transfrom(imageB)

        return imageA, imageB

    def __len__(self):
        return self.length
