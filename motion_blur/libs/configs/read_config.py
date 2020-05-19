import yaml


class parse_config:
    def __init__(self, config_path):
        """ 
            This class is just used as a container to avoid using dictionaries (less readable)
        """

        with open(config_path) as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)

        self.train_dataset_path = config_dict["TRAIN"]["TRAIN_DATASET_PATH"]
        self.save_path = config_dict["TRAIN"]["SAVE_PATH"]
        self.n_epoch = config_dict["TRAIN"]["N_EPOCH"]
        self.lr = config_dict["TRAIN"]["LR"]
        self.L_max = config_dict["TRAIN"]["L_max"]
        self.L_min = config_dict["TRAIN"]["L_min"]
        self.loss_period = config_dict["TRAIN"]["LOSS_PERIOD"]
        self.mini_batch_size = config_dict["TRAIN"]["MINI_BATCH_SIZE"]

        self.sharp_val_dataset_path = config_dict["VALIDATION"]["SHARP_VAL_DATASET_PATH"]
        self.blurred_val_dataset_path = config_dict["VALIDATION"]["BLURRED_VAL_DATASET_PATH"]
        self.validation_period = config_dict["VALIDATION"]["VALIDATION_PERIOD"]

        self.test_dataset_path = config_dict["TEST"]["TEST_DATASET_PATH"]
        self.weight_path = config_dict["TEST"]["WEIGHT_PATH"]
