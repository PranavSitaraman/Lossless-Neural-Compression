import yaml
from pathlib import Path

class Config:
    def __init__(self, config_file='config.yaml'):
        self.config_file = config_file
        self._load_config()

    def _load_config(self):
        with open(self.config_file, 'r') as f:
            self.config = yaml.safe_load(f)

    @property
    def data_root(self):
        return self.config['data']['root']

    @property
    def batch_size(self):
        return self.config['data']['batch_size']

    @property
    def training_size(self):
        return self.config['data']['training_size']

    @property
    def total_size(self):
        return self.config['data']['total_size']

    @property
    def image_size(self):
        return self.config['data']['image_size']

    @property
    def experiment_root(self):
        return Path(self.config['experiments']['root'])

    @property
    def model_path(self):
        return self.experiment_root / self.config['experiments']['model_path']
    
    @property
    def analysis_path(self):
        return self.experiment_root / self.config['experiments']['analysis_path']

    @property
    def results_dir(self):
        return self.experiment_root / self.config['experiments']['results_dir']

    @property
    def logs_dir(self):
        return self.experiment_root / self.config['experiments']['logs_dir']

    @property
    def compression_algorithm(self):
        return self.config['compression']['algorithm']

    @property
    def compression_quality(self):
        return self.config['compression']['quality']

    @property
    def training_epochs(self):
        return self.config['training']['epochs']

    @property
    def learning_rate(self):
        return self.config['training']['learning_rate']

# Create a singleton instance
config = Config()