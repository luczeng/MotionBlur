import wget
import zipfile
from pathlib import Path


if __name__ == "__main__":

    # Input, output paths
    train_url = "http://images.cocodataset.org/zips/train2014.zip"
    val_url = "http://images.cocodataset.org/zips/val2014.zip"
    saving_folder = "datasets"

    Path(saving_folder).mkdir(exist_ok=True)

    # Download and unzip
    train_path = Path(saving_folder) / "train2014"
    if not Path(train_path).exists():
        # wget.download(train_url, saving_folder)

        with zipfile.ZipFile(Path(saving_folder) / "train2014.zip", "r") as zip_ref:
            zip_ref.extractall(saving_folder)

        (Path(saving_folder) / "train2014.zip").unlink()

    val_path = Path(saving_folder) / "val2014"
    if not Path(val_path).exists():
        wget.download(val_url, saving_folder)

        with zipfile.ZipFile(Path(saving_folder) / "val2014.zip", "r") as zip_ref:
            zip_ref.extractall(saving_folder)

        (Path(saving_folder) / "val2014.zip").unlink()
