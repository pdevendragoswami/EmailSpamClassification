import joblib 
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from EmailSpamClassification.components.stage_03_data_transformation import DataTransformation,DataTransformationConfig



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))


    def predict_value(self,data,config:DataTransformationConfig):


        data_transformation_obj = DataTransformation(config=DataTransformationConfig)

        clean_data = data_transformation_obj.get_cleaned_data(data)

        loaded_cv = joblib.load(Path('artifacts/data_transformation/count_vectorizer.joblib'))


        final_data = loaded_cv.transform(clean_data).toarray()

        
        prediction = self.model.predict(final_data)

        print(prediction)

        return prediction
        