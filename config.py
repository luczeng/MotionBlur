import argparse

IMG_PATH = "tests/lena.jpeg"


def get_config(argv):

    parser = argparse.ArgumentParser(
        description="Script that performs either training or testing. If testing, we only test on the Lena image."
    )
    parser.add_argument(
        "-s",
        "--save_model",
        type=int,
        default=0,
        help="Save weights or not. If training, put 1 to save model and specify saving path.",
    )
    parser.add_argument("-l", "--load_model", type=int, default=0, help="Load model. Put 1 for testing")
    parser.add_argument("-p", "--path", type=str, help="Path to model file.", default="empty")

    cfg, unparsed = parser.parse_known_args()
    cfg.img_path = IMG_PATH

    return cfg
