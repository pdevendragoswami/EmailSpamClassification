import os
import numpy as np
import pandas as pd
import joblib
from urllib.parse import urlparse
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from EmailSpamClassification.utils.utils import save_json
from EmailSpamClassification.entity.config_entity import ModelEvaluationConfig
from EmailSpamClassification.config.configuration import ConfigurationManager


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self,actual, pred):
        accuracy_value = accuracy_score(actual,pred)

        #cf_matrix = confusion_matrix(actual,pred)

        #cls_report = classification_report(actual,pred)
        
        return accuracy_value #, cf_matrix, cls_report

    
    def save_results(self):

        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        test_x = test_data.drop([self.config.target_column], axis=1)
        test_y = test_data[[self.config.target_column]]
        
        predicted_class = model.predict(test_x)

        (accuracy) = self.eval_metrics(test_y, predicted_class)
        
        # Saving metrics as local
        scores = {"Accuracy": accuracy,}# "Confusion_matrix": cf_matrix, "Classification_Report": cls_report}
        save_json(path=Path(self.config.metric_file_name), data=scores)



'''
config = ConfigurationManager()
model_evaluation_config = config.get_model_evaluation_config()
model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
model_evaluation_config.save_results()
'''