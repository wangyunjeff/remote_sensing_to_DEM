import os

import torch

import config.config as config

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
    optimizer.load_state_dict(checkpoint["optimizer"],)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr