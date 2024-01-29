import os
import joblib
import pandas as pd
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from EmailSpamClassification import logger
from EmailSpamClassification.entity.config_entity import ModelTrainerConfig
from EmailSpamClassification.config.configuration import ConfigurationManager



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config


    
    def train(self):
        train_data = pd.read_csv(self.config.train_data_path)
        test_data = pd.read_csv(self.config.test_data_path)


        train_x = train_data.drop([self.config.target_column], axis=1)
        test_x = test_data.drop([self.config.target_column], axis=1)
        train_y = train_data[[self.config.target_column]]
        test_y = test_data[[self.config.target_column]]


        rfc = RandomForestClassifier(random_state=self.config.random_state)
        rfc.fit(train_x, train_y)

        joblib.dump(rfc, os.path.join(self.config.root_dir, self.config.model_name))

'''
config = ConfigurationManager()
model_trainer_config = config.get_model_trainer_config()
model_trainer_config = ModelTrainer(config=model_trainer_config)
model_trainer_config.train()
'''