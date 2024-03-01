import torch
import torch.nn as nn
import torch.optim as optim
import config
from dataset import ShoesDataset
from Discriminator import discriminator
from Generator import Generator
from tqdm import tqdm
from torch.utils.data import DataLoader
from skimage.metrics import structural_similarity as ssim


def train(
        disc,gen,loader,opt_disc,opt_gen,L1_loss,bce):

   loop=tqdm(loader,leave=True)
   
   for idx,(x,y) in enumerate(loop):
    x=x.to(config.DEVICE)
    y=y.to(config.DEVICE)

    y_fake=gen(x)
    D_real=disc(x,y)
    D_real_loss=bce(D_real,torch.ones_like(D_real))
    D_fake=disc(x,y_fake.detach())
    D_fake_loss=bce(D_fake,torch.zeros_like(D_fake))
    D_loss=(D_real_loss + D_fake_loss)/2

    opt_disc.zero_grad()
    D_loss.backward()
    opt_disc.step()

    D_fake=disc(x,y_fake)
    G_fake_loss=bce(D_fake,torch.ones_like(D_fake))
    L1=L1_loss(y_fake,y)*config.L1_LAMBDA
    G_loss= G_fake_loss + L1

    opt_gen.zero_grad()
    G_loss.backward()
    opt_gen.step()

    if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )
    ssim_value = compute_ssim(y, y_fake)

        # Print losses and SSIM
    loop.set_postfix(
            D_real=torch.sigmoid(D_real).mean().item(),
            D_fake=torch.sigmoid(D_fake).mean().item(),
            SSIM=ssim_value
        )


def compute_ssim(real_img, generated_img):
    real_img = real_img.permute(0, 2, 3, 1).cpu().numpy()  # Change to HWC format
    generated_img = generated_img.permute(0, 2, 3, 1).cpu().detach().numpy()  # Change to HWC format

    ssim_value = 0
    for i in range(real_img.shape[0]):
        ssim_value += ssim(real_img[i], generated_img[i], multichannel=True)

    return ssim_value / real_img.shape[0]

def main():
    disc=discriminator().to(config.DEVICE)
    gen=Generator(in_channels=3).to(config.DEVICE)
    opt_disc=optim.Adam(disc.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    opt_gen=optim.Adam(gen.parameters(),lr=config.LEARNING_RATE,betas=(0.5,0.999))
    BCE=nn.BCEWithLogitsLoss()
    L1_loss=nn.L1Loss()

    train_dataset=ShoesDataset(root_dir=config.TRAIN_DIR)
    train_loader=DataLoader(
       train_dataset,
       batch_size=config.BATCH_SIZE,
       shuffle=True,
       num_workers=config.NUM_WORKERS
    
    )
    val_dataset=ShoesDataset(root_dir=config.VAL_DIR)
    val_loader=DataLoader(val_dataset,batch_size=1,shuffle=False)

    for epoch in range(config.NUM_EPOCHS):
       train(
          disc,gen,train_loader,opt_disc,opt_gen,L1_loss,BCE)
       
       


if __name__ == "__main__":
    main()