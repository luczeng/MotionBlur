import yaml


class parse_config:
    def __init__(self, config_path):
        """
            This class is just used as a container to avoid using dictionaries (less readable)
        """

        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        # General
        self.config_path = config_path

        # Network parameters
        self.n_layers = config_dict["NET"]["N_LAYERS"]
        self.n_sublayers = config_dict["NET"]["N_SUBLAYERS"]
        self.n_features_first_layer = config_dict["NET"]["N_FEATURES_FIRST_LAYER"]
        self.as_gray = bool(config_dict["NET"]["AS_GRAY"])
        self.regression = bool(config_dict["NET"]["REGRESSION"])

        # Train parameters
        self.train_dataset_path = config_dict["TRAIN"]["TRAIN_DATASET_PATH"]
        self.save_path = config_dict["TRAIN"]["SAVE_PATH"]
        self.n_epoch = config_dict["TRAIN"]["N_EPOCH"]
        self.lr = config_dict["TRAIN"]["LR"]
        self.L_max = config_dict["TRAIN"]["L_max"]
        self.L_min = config_dict["TRAIN"]["L_min"]
        self.n_angles = config_dict["TRAIN"]["n_angles"]
        self.n_lengths = config_dict["TRAIN"]["n_lengths"]
        self.loss_period = config_dict["TRAIN"]["LOSS_PERIOD"]
        self.mini_batch_size = config_dict["TRAIN"]["MINI_BATCH_SIZE"]
        self.small_dataset = bool(config_dict["TRAIN"]["SMALL_DATASET"])
        self.saving_epoch = config_dict["TRAIN"]["SAVING_EPOCH"]
        self.load_checkpoint = bool(config_dict["TRAIN"]["LOAD_CHECKPOINT"])

        # Validation parameters
        self.use_validation = bool(config_dict["VALIDATION"]["USE_VALIDATION"])
        self.sharp_val_dataset_path = config_dict["VALIDATION"]["SHARP_VAL_DATASET_PATH"]
        self.blurred_val_dataset_path = config_dict["VALIDATION"]["BLURRED_VAL_DATASET_PATH"]
        self.validation_period = config_dict["VALIDATION"]["VALIDATION_PERIOD"]
        self.val_small_dataset_path = config_dict["VALIDATION"]["VAL_SMALL_DATASET_PATH"]
        self.val_n_angles = config_dict["VALIDATION"]["VAL_N_ANGLES"]

        # Test parameters
        self.test_dataset_path = config_dict["TEST"]["TEST_DATASET_PATH"]
        self.weight_path = config_dict["TEST"]["WEIGHT_PATH"]

        # Mlflow
        self.log_weights = bool(config_dict["MLFLOW"]["LOG_WEIGHTS"])
