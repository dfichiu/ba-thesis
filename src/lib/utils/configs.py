import yaml


class InferenceConfig:
    def __init__(
        self,
        sentences,
        tracked_tokens
    ):
        self.sentences = sentences
        self.tracked_tokens = tracked_tokens

class TrainConfig:
     def __init__(
         self,
         data_path,
         chunk_sizes,
         epochs
     ):
        self.data_path = data_path
        self.chunk_sizes = chunk_sizes
        self.epochs = epochs
        

class DSDMConfig:
    def __init__(
        self,
        address_size,
        ema_time_period,
        learning_rate_update,
        temperature,
        normalize
        
    ):
        self.address_size = address_size
        self.ema_time_period = ema_time_period
        self.learning_rate_update = learning_rate_update
        self.temperature = temperature
        self.normalize = normalize
        

class Config:
    def __init__(self, config: dict):
        self.experiment_title = config.get(
            'experiment_title', " 'Default' "
        ) 
        self.experiment_type = config.get(
            'experiment_type', "run"
        ) 

        # Set hypervector dimension.
        self.dim = config.get(
            'dim', 2000
        ) 
        
        self.initial_training = self._get_training_config(config, 'initial_training')
        
        self.training = self._get_training_config(config, training_key='training')
       
        self.DSDM = self._get_DSDM_config(config)
        
        self.inference = self._get_inference_config(config)
        
        self._check_experiment_parameters()
        
    def _get_DSDM_config(self, config: dict) -> DSDMConfig:
        if config.get('DSDM'):
            ema_time_period = config['DSDM'].get('ema_time_period')
            learning_rate_update = config['DSDM'].get('learning_rate_update')
            temperature = config['DSDM'].get('temperature')
            normalize = config['DSDM'].get('normalize')
        #else:
        #    # Set the values of the default experiment.
        #    ema_time_period = 
        #    learning_rate_update = 
        #    temperature = 
        #    normalize = 

        return DSDMConfig(
            self.dim,
            ema_time_period,
            learning_rate_update,
            temperature,
            normalize
        )

    def _get_training_config(self, config: dict, training_key: str) -> TrainConfig:
        if config.get(training_key):
            data_path = config[training_key].get('data_path')
            chunk_sizes = config[training_key].get('chunk_sizes')
            epochs = config[training_key].get('epochs')
        #else:
        #    # Set the values of the default experiment.
        #    data_path = ""
        #    chunk_size = [1, 2, 3]
        #    epochs = None

        return TrainConfig(
            data_path,
            chunk_sizes,
            epochs
        )

    def _get_inference_config(self, config: dict) -> InferenceConfig:
        if config.get('inference'):
            sentences = config['inference'].get('sentences')
            tracked_tokens = config['inference'].get('tracked_tokens')
        #else:
        #    # Set the values of the default experiment.
        #    inference_sentences = 
        #    tracked_tokens = 

        return InferenceConfig(
            sentences,
            tracked_tokens
        )
    
    def _check_experiment_parameters(self):
        """
        
        """
        pass

    @staticmethod
    def from_file(config_path):
        try:
            # Read YAML file.
            with open(config_path) as stream:
                config = yaml.safe_load(stream)
                return Config(config)
        except FileNotFoundError:
            print(f"File '{config_path}' not found.")
            return None
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            return None

        