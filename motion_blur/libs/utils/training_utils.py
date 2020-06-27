def print_info(running_loss, epoch, idx, dataset_len, config):
    """
        Convenience function to print info each specified period. Resets the running loss.

        :param running_loss
        :param epoch
        :param idx iteration index (for one epoch)
        :param dataset_len number of points in the dataset
        :param config
        :return running_loss, iterations resetted to 0
        TODO: Move to other file (training_utils.py?)
    """

    # Prints loss
    print(
        "Epoch : %d/%d, iteration: %d/%5d || loss: %.3f"
        % (epoch, config.n_epoch, idx + 1, dataset_len / config.mini_batch_size, running_loss / config.loss_period,)
    )

    return 0.0


def print_info_small_dataset(running_loss, iterations, epoch, idx, dataset_len, config):
    """
        Convenience function to print info each specified period. Resets the running loss.
        For small dataset scripts.

        :param running_loss
        :param iterations number of iterations to divide the loss with
        :param epoch
        :param idx iteration index (for one epoch)
        :param dataset_len number of points in the dataset
        :param config
        :return running_loss, iterations resetted to 0
        TODO: Move to other file (training_utils.py?)
    """

    # Prints loss
    print("Epoch: %d/%d || loss: %.3f" % (epoch, config.n_epoch, running_loss / config.loss_period,))

    return 0.0, 0
