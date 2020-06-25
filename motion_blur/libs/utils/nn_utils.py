import torch
import mlflow
from pathlib import Path
from torchsummary import summary


def print_training_info(net, img_size):
    print("\nNetwork information: \n")
    print(net, "\n")
    summary(net, (3, img_size[0], img_size[1]))
    print("\n")


def log_mlflow_param(config):
    """
        Log mlflow metrics.
        Logs the whole config file and some specific parameters that are to be seen at first by the user
        :param config parse_config file
    """

    mlflow.log_artifact(config.config_path)
    mlflow.log_param("lr", config.lr)
    mlflow.log_param("dataset_name", Path(config.train_dataset_path).name)
    mlflow.log_param("n_layers", config.n_layers)
    mlflow.log_param("n_sublayers", config.n_sublayers)
    mlflow.log_param("n_features", config.n_features_first_layer)


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
