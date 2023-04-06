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
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifact','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
    

