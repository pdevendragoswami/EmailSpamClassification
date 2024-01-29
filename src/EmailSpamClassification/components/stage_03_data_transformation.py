import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from EmailSpamClassification import logger
from EmailSpamClassification.entity.config_entity import DataTransformationConfig
from EmailSpamClassification.config.configuration import ConfigurationManager
import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
import joblib

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    
    def get_cleaned_data(self,dataframe):
        try:
            ps = PorterStemmer()
            corpus = []
            for i in range(0, len(dataframe)):
                review = re.sub('[^a-zA-Z]', ' ', dataframe['text'][i])
                review = review.lower()
                review = review.split()
                review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
                review = ' '.join(review)

                corpus.append(review)
            return corpus
        
        except Exception as e:
            raise e


    def train_test_spliting(self):

        data = pd.read_csv(self.config.data_path)
        #print(data.head())

        corpus = self.get_cleaned_data(data)
        ##print(corpus)
        #print(self.config.target_column)
        target_feature = data[[self.config.target_column]]
        #print(target_feature)

        # Split the data into training and test sets. (0.75, 0.25) split.
        X_train, X_test, y_train, y_test = train_test_split(corpus,target_feature,test_size=0.2, random_state=42)

        cv = CountVectorizer(max_features=2500)

        
        # Fit and transform with CountVectorizer
        X_train = cv.fit_transform(X_train).toarray()
        X_test = cv.transform(X_test).toarray()

        joblib.dump(cv, os.path.join(self.config.root_dir, self.config.cv_name))

        # Get feature names from CountVectorizer
        cols = list(cv.get_feature_names_out())

        columns_name = cols + [self.config.target_column]
        #print(columns_name)

        #print(X_train.shape)
        #print(X_test.shape)
        #print(y_train.shape)
        #print(y_test.shape)
     

        # Create DataFrames
        train_data = np.column_stack((X_train, y_train))
        test_data = np.column_stack((X_test, y_test))

        train_data = pd.DataFrame(train_data, columns= columns_name)
        test_data = pd.DataFrame(test_data, columns= columns_name)


        # Save DataFrames to CSV
        train_data.to_csv(os.path.join(self.config.root_dir, "train.csv"), index=False)
        test_data.to_csv(os.path.join(self.config.root_dir, "test.csv"), index=False)

        logger.info("Splited data into training and test sets")
        logger.info(train_data.shape)
        logger.info(test_data.shape)

        print(train_data.shape)
        print(test_data.shape)
        
        
'''
config = ConfigurationManager()
data_transformation_config = config.get_data_transformation_config()
data_transformation = DataTransformation(config=data_transformation_config)
data_transformation.train_test_spliting() 
'''