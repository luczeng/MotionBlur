from motion_blur.libs.metrics.metrics import create_validation_dataset, run_validation
from motion_blur.libs.configs.read_config import parse_config


if __name__ == "__main__":
    """
        Creates the validation dataset by blurring images and saving kernel information
        Change validation path in config file if needed

        TODO: add argparse
        TODO: parralelize process
    """

    # config
    path_to_config = "motion_blur/libs/configs/config_motionnet.yml"
    config = parse_config(path_to_config)

    # Create val dataset and saves it to disk
    create_validation_dataset(config)
