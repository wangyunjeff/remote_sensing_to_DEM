import os

import torch
import matplotlib.pyplot as plt
import config.config as config


def unnormalized(img, mean, std):
    img = img * torch.tensor(std).view(-1, 1, 1).to(config.DEVICE) + torch.tensor(mean).view(-1, 1, 1).to(config.DEVICE)
    return img

def save_some_examples(gen, val_loader, epoch, folder):
    for i, (x, y) in enumerate(val_loader):
        if i == 0:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            gen.eval()
            with torch.no_grad():
                y_fake = gen(x)
                x = unnormalized(img=x, mean=[0.2895, 0.3111, 0.2108], std=[0.1522, 0.1278, 0.1191])
                y_fake = unnormalized(img=y_fake, mean=[2532.5225], std=[1147.7783]) / 8000
                if epoch == 0:
                    fig, axes = plt.subplots(3, 3)
                    ax = axes.ravel()
                    for j in range(9):
                        ax[j].imshow(x[j].squeeze().permute(1, 2, 0).detach().cpu().numpy())
                        ax[j].axis('off')
                    plt.tight_layout()
                    plt.savefig(folder + '/input.png')
                    plt.close()

                    fig, axes = plt.subplots(3, 3)
                    ax = axes.ravel()
                    for j in range(9):
                        ax[j].imshow(y[j].squeeze().detach().cpu().numpy())
                        ax[j].axis('off')
                    plt.tight_layout()
                    plt.savefig(folder + '/label.png')
                    plt.close()

                    fig, axes = plt.subplots(3, 3)
                    ax = axes.ravel()
                    for j in range(9):
                        ax[j].imshow(y_fake[j].squeeze().detach().cpu().numpy())
                        ax[j].axis('off')
                    plt.tight_layout()
                    plt.savefig(folder + '/fake_{}.png'.format(epoch))
                    plt.close()

                else:
                    fig, axes = plt.subplots(3, 3)
                    ax = axes.ravel()
                    for j in range(9):
                        ax[j].imshow(y_fake[j].squeeze().detach().cpu().numpy())
                        ax[j].axis('off')
                    plt.tight_layout()
                    plt.savefig(folder + '/fake_{}.png'.format(epoch))
                    plt.close()

            gen.train()

        else:
            x = x.to(config.DEVICE)
            y = y.to(config.DEVICE)

            gen.eval()
            with torch.no_grad():
                y_fake = gen(x)
                x = unnormalized(img=x, mean=[0.2895, 0.3111, 0.2108], std=[0.1522, 0.1278, 0.1191])
                y_fake = unnormalized(img=y_fake, mean=2532.5225, std=1147.7783) / 8000
                fig, axes = plt.subplots(3, 3)
                ax = axes.ravel()
                for j in range(9):
                    ax[j].imshow(x[j].squeeze().permute(1, 2, 0).detach().cpu().numpy())
                    ax[j].axis('off')
                plt.tight_layout()
                plt.savefig(folder + '/input_{}.png'.format(i))
                plt.close()

                fig, axes = plt.subplots(3, 3)
                ax = axes.ravel()
                for j in range(9):
                    ax[j].imshow(y[j].squeeze().detach().cpu().numpy())
                    ax[j].axis('off')
                plt.tight_layout()
                plt.savefig(folder + '/label_{}.png'.format(i))
                plt.close()

                fig, axes = plt.subplots(3, 3)
                ax = axes.ravel()
                for j in range(9):
                    ax[j].imshow(y_fake[j].squeeze().detach().cpu().numpy())
                    ax[j].axis('off')
                plt.tight_layout()
                plt.savefig(folder + '/fake_{}.png'.format(i))
                plt.close()

            gen.train()

def save_checkpoint(model, optimizer, filepath='../runs/weights/checkpoint.pt'):
    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))
    print("-> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("-> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"], )

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
