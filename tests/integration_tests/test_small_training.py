from driver_scripts.main_train import run_train


class mock_args:
    def __init__():
        pass


def test_train_classification_network():

    config_path = "tests/test_data/configs/mock_config_classification.yml"

    args = mock_args
    args.config_path = config_path

    # Launch training
    run_train(args)


def test_train_estimation_network():

    config_path = "tests/test_data/configs/mock_config_estimation.yml"

    args = mock_args
    args.config_path = config_path

    # Launch training
    run_train(args)
