from motion_blur.libs.data.datasets_preparation import prepare_reds_dataset
import argparse
from pathlib import Path


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Moves reds images into two folders called val_reds and train_reds")
    parser.add_argument("-v", "--path_to_val")
    parser.add_argument("-t", "--path_to_train")
    parser.add_argument("-o", "--output_folder_path")
    args = parser.parse_args()

    path_to_output_val = Path(args.output_folder_path) / "val_reds"
    path_to_output_train = Path(args.output_folder_path) / "train_reds"

    prepare_reds_dataset(args.path_to_val, args.path_to_train, str(path_to_output_val), str(path_to_output_train))
