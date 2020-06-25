from motion_blur.libs.metrics.metrics_small_dataset import evaluate_one_image_classification
from motion_blur.libs.nn.motion_net import MotionNet
from motion_blur.libs.configs.read_config import parse_config
import torch


def test_small_dataset_classification_metrics_gray():

    # Parameters:
    config_path = "tests/test_data/configs/mock_config_classification.yml"
    img_path = "tests/test_data/test_img/lena.jpeg"
    net_type = torch.FloatTensor
    n_angles = 2
    L_min = 5
    L_max = 5
    as_gray = True

    # Configs
    cfg = parse_config(config_path)

    net = MotionNet(
        cfg.n_layers, cfg.n_sublayers, cfg.n_features_first_layer, [0, 0], as_gray, cfg.regression, n_angles, 1
    )

    angle_loss = evaluate_one_image_classification(net, img_path, net_type, n_angles, L_min, L_max, as_gray)

    assert angle_loss.item() == 90


def test_small_dataset_classification_metrics_rgb():

    # Parameters:
    config_path = "tests/test_data/configs/mock_config_classification.yml"
    img_path = "tests/test_data/test_img/lena.jpeg"
    net_type = torch.FloatTensor
    n_angles = 2
    L_min = 5
    L_max = 5
    as_gray = False

    # Configs
    cfg = parse_config(config_path)

    net = MotionNet(
        cfg.n_layers, cfg.n_sublayers, cfg.n_features_first_layer, [0, 0], as_gray, cfg.regression, n_angles, 1
    )

    angle_loss = evaluate_one_image_classification(net, img_path, net_type, n_angles, L_min, L_max, as_gray)

    assert angle_loss.item() == 90
