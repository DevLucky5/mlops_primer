import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation, DataTransformationConfig
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
    
    def spinup_data_ingestion(self):
        logging.info('Data ingestion started.')
        try:
            df = pd.read_csv('notebook\data\stud.csv')
            logging.info('Dataset fetch completed.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)
            logging.info('Dataset is saved. Preparing train and test data...')

            train, test = train_test_split(df, test_size = 0.2, random_state=4)
            logging.info('Train and Test datasets are prepared.')

            train.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test.to_csv(self.ingestion_config.test_data_path, index = False, header = True)
            logging.info('Train and Test datasets are saved.')

            logging.info('Data Ingestion process completed successfully.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.spinup_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.transform(train_data, test_data)

    model_trainer = ModelTrainer()
    model_score = model_trainer.train(train_arr, test_arr)
    print(model_score)

    