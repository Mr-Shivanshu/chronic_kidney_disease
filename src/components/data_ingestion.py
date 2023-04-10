from src.logger import logging
from src.exception import CustomException
import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import  DataTransformationConfig
from src.components.model_selection import ModelTrainer
from src.components.model_selection import ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    train_data_path=os.path.join('artifacts','train.csv')
    test_data_path=os.path.join('artifacts','test.csv')
    raw_data_path=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info('Data ingestion is initiated')
        try:
            
            df=pd.read_csv("Notebook\data\coronic1.csv")
            logging.info('The file is being read successfully')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            logging.info('train test split initiated')
            train_set,test_set=train_test_split(df,random_state=42,test_size=0.2)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info('Train test split completed')

            return(self.ingestion_config.train_data_path,
                   self.ingestion_config.test_data_path
                   )
        except Exception as e:
            logging.info(f"these are the exception {e}")
            raise CustomException(e,sys)

#if __name__=="__main__":
    #obj=DataIngestion()
    #train_data,test_data=obj.initiate_data_ingestion()
    #data_transformation=DataTransformation()
    #train_arr,test_arr=data_transformation.initiate_data_transformation(train_data,test_data)
    #modeltrainer=ModelTrainer()
    #print(modeltrainer.initiate_model_trainer(train_arr,test_arr))
    




            




