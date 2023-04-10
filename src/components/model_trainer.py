import os
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    model_filepath = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
    
    def train(self, train_arr, test_arr):
        try:
            logging.info('Splitting training and test input data...')
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            models = {
                'RF': RandomForestRegressor(),
                'DT': DecisionTreeRegressor(),
                'GB': GradientBoostingRegressor(),
                'LR': LinearRegression(),
                'KNN': KNeighborsRegressor(),
                'XGB': XGBRegressor(),
                'ADA': AdaBoostRegressor(),
            }

            model_report:dict = evaluate_models(x_train=x_train, y_train=y_train, 
                    x_test= x_test, y_test=y_test, 
                    models = models
                )

            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise Exception('No best model found.')
            
            logging.info('Best model found.')

            save_object(
                file_path= self.config.model_filepath,
                obj = best_model
            )

            y_pred = best_model.predict(x_test)
            r2score = r2_score(y_test, y_pred)
            
            return r2score

        except Exception as e:
            raise CustomException(e, sys)