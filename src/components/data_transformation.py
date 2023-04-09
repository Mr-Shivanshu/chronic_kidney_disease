from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from dataclasses import dataclass
import os
import sys
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    def get_data_transformer_object(self):
        try:
            numerical_columns=['Age', 'Blood Pressure', 'Specific Gravity', 'Albumin', 'Sugar',
                    'Blood Glucose Random', 'Blood Urea', 'Serum Creatinine', 'Sodium',
                        'Potassium', 'Hemoglobin', 'Packed Cell Volume',
                    'White Blood Cell Count', 'Red Blood Cell Count']
            categorical_columns=['Red Blood Cells', 'Pus Cell', 'Pus Cell clumps', 'Bacteria',
                        'Hypertension', 'Diabetes Mellitus', 'Coronary Artery Disease',
                            'Appetite', 'Pedal Edema', 'Anemia']

            numerical_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                               ("scaler",StandardScaler())])
            
            logging.info("numerical column scaling is completed")

            categorical_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),
                                                 ('ordinal_encoder',OrdinalEncoder()),("scaler",StandardScaler())])
            
            logging.info("Categorical column encoding is completed")

            preprocessor=ColumnTransformer(
                [("num_pipeline",numerical_pipeline,numerical_columns),("cat_pipeline",categorical_pipeline,categorical_columns)]
            )

            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("reading of train and test data is completed")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj=self.get_data_transformer_object()
            target_column_name='Class'
            
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)

            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(f"Applying transforming object on train and test data frame")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            logging.info('completed')
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            

            train_arr=np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]
            logging.info(f"saved processing object ")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            return (
                    train_arr,
                    test_arr
                    
                )

                
        except Exception as e:
            logging.info(e)
            return CustomException(e,sys)




                

