import os
from importlib.machinery import SourceFileLoader

class Configuration():
    def __init__(self, config_path):
        self.config_path = config_path

    def load(self):
        # load experiment config file
        cf = SourceFileLoader('config', self.config_path).load_module()

        # create experiment paths
        cf.train_output_path = os.path.join(cf.experiments_path, cf.experiment_name, 'unet_train')
        cf.test_output_path = os.path.join(cf.experiments_path, cf.experiment_name, 'unet_test')

        if not os.path.exists(cf.train_output_path):
            os.makedirs(cf.train_output_path)
        if not os.path.exists(cf.test_output_path):
            os.makedirs(cf.test_output_path)
            os.makedirs(os.path.join(cf.test_output_path, 'predictions'))
            os.makedirs(os.path.join(cf.test_output_path, 'regions'))

        return cf