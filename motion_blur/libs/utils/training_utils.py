def print_info(running_loss, epoch, idx, dataset_len, config):
    """ 
        Convenience function to print info each specified period. Resets the running loss.
        :param running_loss
        :param epoch
        :param idx iteration index (for one epoch)
        :param dataset_len number of points in the dataset
        :param config
        TODO: Move to other file (training_utils.py?)
    """
    # Prints loss
    print(
        "[%d  %d epoch] [%d, %5d] loss: %.3f"
        % (epoch, config.n_epoch, idx + 1, dataset_len / config.mini_batch_size, running_loss / config.loss_period,)
    )
    running_loss = 0.0
