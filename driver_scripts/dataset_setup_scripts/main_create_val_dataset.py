from motion_blur.libs.metrics.metrics import create_validation_dataset
from motion_blur.libs.configs.read_config import parse_config
import argparse


if __name__ == "__main__":
    """
        Creates the validation dataset by blurring images and saving kernel information
        Change validation path in config file if needed

        TODO: add argparse
        TODO: parralelize process
    """

    parser = argparse.ArgumentParser(description="Prepare validation dataset by blurring it")
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()

    # config
    config = parse_config(args.config)

    # Create val dataset and saves it to disk
    create_validation_dataset(config)
