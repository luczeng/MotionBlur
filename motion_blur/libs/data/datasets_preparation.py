from pathlib import Path
import shutil


def prepare_reds_dataset(path_to_val: str, path_to_train: str, output_val_path: str, output_train_path: str):
    """
        Extracts all reds training and validation reds images in one folder.
        Performs sampling as required. Moves the data for speed reaons

        :param path_to_val path to validation dataset
        :param path_to_train path to train dataset
        :param output_val_path path to output_val_path
        :param output_train_path path to output_train_path
    """

    assert Path(path_to_val).exists(), "Validation path does not point to correct folder"
    assert Path(path_to_train).exists(), "Training path does not point to correct folder"

    val_rootdir = Path(path_to_val)
    train_rootdir = Path(path_to_train)

    Path(output_val_path).mkdir(parents=True, exist_ok=True)
    Path(output_train_path).mkdir(parents=True, exist_ok=True)

    val_list = [f for f in val_rootdir.glob("**/*") if f.is_file()]
    train_list = [f for f in train_rootdir.glob("**/*") if f.is_file()]

    out_val_list = [Path(output_val_path) / (f.parent.name + "_" + f.name) for f in val_list]
    out_train_list = [Path(output_train_path) / (f.parent.name + "_" + f.name) for f in train_list]

    for idx in range(len(out_val_list)):
        shutil.move(val_list[idx], out_val_list[idx])

    for idx in range(len(out_train_list)):
        shutil.move(train_list[idx], out_train_list[idx])
