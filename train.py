import os
import torch
import torch.nn as nn
import torch.optim as optim
from data import generate_loader
from option import get_option
from model import Generator, Discriminator
from tqdm import tqdm


class Solver():
    def __init__(self, opt):
        self.opt = opt
        self.img_size = opt.input_size
        self.dev = torch.device("cuda:{}".format(
            opt.gpu) if torch.cuda.is_available() else "cpu")
        print("device: ", self.dev)

        self.generator_A = Generator(
            img_channels=3, num_residuals=9).to(self.dev)
        self.generator_B = Generator(
            img_channels=3, num_residuals=9).to(self.dev)

        self.discriminator_A = Discriminator(in_channels=3).to(self.dev)
        self.discriminator_B = Discriminator(in_channels=3).to(self.dev)

        if opt.pretrained:
            load_path = os.path.join(opt.chpt_root, "Gen_A.pt")
            self.generator_A.load_state_dict(torch.load(load_path))

            load_path = os.path.join(opt.chpt_root, "Gen_B.pt")
            self.generator_B.load_state_dict(torch.load(load_path))

            load_path = os.path.join(opt.chpt_root, "Disc_A.pt")
            self.discriminator_A.load_state_dict(torch.load(load_path))

            load_path = os.path.join(opt.chpt_root, "Disc_B.pt")
            self.discriminator_B.load_state_dict(torch.load(load_path))

        if opt.multigpu:
            self.generator_A = nn.DataParallel(
                self.generator_A, device_ids=self.opt.device_ids).to(self.dev)
            self.generator_B = nn.DataParallel(
                self.generator_B, device_ids=self.opt.device_ids).to(self.dev)
            self.discriminator_A = nn.DataParallel(
                self.discriminator_A, device_ids=self.opt.device_ids).to(self.dev)
            self.discriminator_B = nn.DataParallel(
                self.discriminator_B, device_ids=self.opt.device_ids).to(self.dev)

        self.l1_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()

        self.optimizer_G = optim.Adam(list(self.generator_A.parameters())+list(self.generator_B.parameters()),
                                      lr=opt.lr, betas=(opt.b1, opt.b2))
        self.optimizer_D = optim.Adam(list(self.discriminator_A.parameters())+list(self.discriminator_B.parameters()),
                                      lr=opt.lr, betas=(opt.b1, opt.b2))

        self.train_loader = generate_loader(opt)
        print("train set ready")

    def fit(self):
        opt = self.opt
        print("start training")

        for epoch in range(opt.n_epoch):
            loop = tqdm(self.train_loader)

            for i, (img_A, img_B) in enumerate(loop):
                img_A = img_A.to(self.dev)
                img_B = img_B.to(self.dev)

                # train Discriminator A
                fake_A = self.generator_A(img_B)
                D_A_real = self.discriminator_A(img_A)
                D_A_fake = self.discriminator_A(fake_A.detach())

                D_A_real_loss = self.mse_loss(
                    D_A_real, torch.ones_like(D_A_real))
                D_A_fake_loss = self.mse_loss(
                    D_A_fake, torch.zeros_like(D_A_fake))
                D_A_loss = D_A_real_loss + D_A_fake_loss

                # train Discriminator B
                fake_B = self.generator_B(img_A)
                D_B_real = self.generator_B(img_B)
                D_B_fake = self.generator_B(fake_B.detach())

                D_B_real_loss = self.mse_loss(
                    D_B_real, torch.ones_like(D_B_real))
                D_B_fake_loss = self.mse_loss(
                    D_B_fake, torch.zeros_like(D_B_fake))
                D_B_loss = D_B_real_loss + D_B_fake_loss

                D_loss = (D_A_loss + D_B_loss) / 2

                self.optimizer_D.zero_grad()
                D_loss.backward()
                self.optimizer_D.step()

                # train Generator A
                D_A_fake = self.discriminator_A(fake_A)
                D_B_fake = self.discriminator_B(fake_B)
                G_A_loss = self.mse_loss(D_A_fake, torch.ones_like(D_A_fake))
                G_B_loss = self.mse_loss(D_B_fake, torch.ones_like(D_B_fake))

                # cycle loss
                cycle_A = self.generator_A(fake_B)
                cycle_B = self.generator_B(fake_A)
                cycle_A_loss = self.l1_loss(img_A, cycle_A)
                cycle_B_loss = self.l1_loss(img_B, cycle_B)

                G_loss = (G_A_loss + G_B_loss +
                          opt.cycle_lambda * (cycle_A_loss + cycle_B_loss))

                self.optimizer_G.zero_grad()
                G_loss.backward()
                self.optimizer_G.step()

            print(
                f"[Epoch {epoch+1}/{opt.n_epoch}] [D loss: {D_loss.item():.6f}] [G loss: {G_loss.item():.6f}]")

            if (epoch+1) % 25 == 0:
                self.save()

    def save(self):
        os.makedirs(os.path.join(self.opt.ckpt_root), exist_ok=True)
        G_A_save_path = os.path.join(self.opt.ckpt_root, "Gen_A.pt")
        G_B_save_path = os.path.join(self.opt.ckpt_root, "Gen_B.pt")

        D_A_save_path = os.path.join(self.opt.ckpt_root, "Disc_A.pt")
        D_B_save_path = os.path.join(self.opt.ckpt_root, "Disc_B.pt")

        torch.save(self.generator_A.state_dict(), G_A_save_path)
        torch.save(self.generator_B.state_dict(), G_B_save_path)
        torch.save(self.discriminator_A.state_dict(), D_A_save_path)
        torch.save(self.discriminator_B.state_dict(), D_B_save_path)


def main():
    opt = get_option()
    torch.manual_seed(opt.seed)
    solver = Solver(opt)
    solver.fit()


if __name__ == "__main__":
    main()
