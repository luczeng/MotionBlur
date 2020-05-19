import mlflow
import numpy as np
import matplotlib.pyplot as plt


def log_loss(loss, loss_period):
    """
    """

    # Loss vector
    x = np.arange(loss_period, loss_period * len(loss), len(loss))

    # Plot loss
    fig = plt.figure(1)
    plt.plot(loss)
    plt.show()

    fig.savefig("img/training_loss.png")
    plt.close(fig)

    mlflow.log_artifact("img/training_loss.png")
