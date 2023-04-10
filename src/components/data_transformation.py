import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_file_path = os.path.join('artifacts', 'tranformer.pkl')

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()
    
    def transfomer(self):
        try:
            num_cols = ['reading_score', 'writing_score']
            cat_cols = [
                'gender', 
                'race_ethnicity', 
                'parental_level_of_education', 
                'lunch', 
                'test_preparation_course'
            ]

            num_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))
                ]
            )

            logging.info('Pipelines created.')

            transformer = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, num_cols),
                    ('cat_pipeline', cat_pipeline, cat_cols)
                ]
            )
            logging.info('Transformer configured.')

            return transformer

        except Exception as e:
            raise CustomException(e, sys)

    def transform(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Train and Test data imported.')

            logging.info('Initiating transformation...')
            transformer = self.transfomer()


            target_col = 'math_score'
            num_cols = ['reading_score', 'writing_score']

            input_train_df = train_df.drop(columns = [target_col], axis= 1)
            target_train_df = train_df[target_col]

            input_test_df = test_df.drop(columns = [target_col], axis= 1)
            target_test_df = test_df[target_col]

            logging.info('Applying transformation...')
            input_train_arr = transformer.fit_transform(input_train_df)
            input_test_arr = transformer.transform(input_test_df)

            train_arr = np.c_[
                input_train_arr, np.array(target_train_df)
            ]
            test_arr = np.c_[
                input_test_arr, np.array(target_test_df)
            ]

            logging.info('Saved the transformer.')

            save_object(
                file_path = self.config.preprocessor_file_path,
                obj = transformer
            )

            return (
                train_arr,
                test_arr,
                self.config.preprocessor_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


