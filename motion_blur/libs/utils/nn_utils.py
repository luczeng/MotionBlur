import torch


def save_checkpoint(state, ckp_path):
    torch.save(state, ckp_path)


def load_checkpoint(ckp_path, model, optimizer):
    """ 
        Updates the model and optimizers to the checkpoint's state
        :param ckp_path path to checkpoint (.pth file)
        :param model
        :param optimizer
        :return: checkpoint's epoch
    """
    checkpoint = torch.load(ckp_path)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["epoch"]


def define_checkpoint(model, optimizer, epoch):
    """
    TODO
    """

    return {"state_dict": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}


def load_model(model_path: str, model):
    model.load_state_dict(checkpoint["state_dict"])
