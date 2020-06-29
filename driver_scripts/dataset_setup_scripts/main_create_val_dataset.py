from motion_blur.libs.utils.evaluation_data_utils import (
    create_validation_dataset_classification,
    create_validation_dataset_regression,
)
from motion_blur.libs.configs.read_config import parse_config
import argparse


if __name__ == "__main__":
    """
        Creates the validation dataset by blurring images and saving kernel information
        Change validation path in config file if needed

        TODO: add argparse
        TODO: parralelize process
        TODO: write test
    """

    parser = argparse.ArgumentParser(description="Prepare validation dataset by blurring it")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-t", "--type", type=str, default="classification", choices=["classification", "regression"])
    args = parser.parse_args()

    # config
    config = parse_config(args.config)

    # Create val dataset and saves it to disk
    if args.type == "classification":
        create_validation_dataset_classification(config)
    elif args.type == "regression":
        create_validation_dataset_regression(config)
    else:
        raise ValueError("Invalid task type")
