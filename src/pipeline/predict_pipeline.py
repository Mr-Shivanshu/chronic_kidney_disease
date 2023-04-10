import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import os
class PredictPipeline:
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print('Before loading')
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            logging.info('load of model and preprocessor object is completed')
            data_scaled=preprocessor.transform(features)
            logging.info('data scaled completed')
            preds=model.predict(data_scaled)
            logging.info(preds)
            return preds
        except Exception as e:

            raise e
        

class CustomData:
    def __init__(self, Age, Blood_Pressure, Specific_Gravity, Albumin, Sugar,
       Red_Blood_Cells, Pus_Cell, Pus_Cell_clumps, Bacteria,
       Blood_Glucose_Random, Blood_Urea, Serum_Creatinine, Sodium,
       Potassium, Hemoglobin, Packed_Cell_Volume,
       White_Blood_Cell_Count, Red_Blood_Cell_Count, Hypertension,
       Diabetes_Mellitus, Coronary_Artery_Disease, Appetite,
       Pedal_Edema, Anemia):
        
       self.Age=Age
       self.Blood_Pressure=Blood_Pressure
       self.Specific_Gravity=Specific_Gravity
       self.Albumin=Albumin
       self.Sugar=Sugar
       self.Red_Blood_Cells=Red_Blood_Cells
       self.Pus_Cell=Pus_Cell
       self.Pus_Cell_clumps=Pus_Cell_clumps
       self.Bacteria=Bacteria
       self.Blood_Glucose_Random=Blood_Glucose_Random
       self.Blood_Urea=Blood_Urea
       self.Serum_Creatinine=Serum_Creatinine
       self.Sodium=Sodium
       self.Potassium=Potassium
       self.Hemoglobin=Hemoglobin
       self.Packed_Cell_Volume=Packed_Cell_Volume
       self.White_Blood_Cell_Count=White_Blood_Cell_Count
       self.Red_Blood_Cell_Count=Red_Blood_Cell_Count
       self.Hypertension=Hypertension
       self.Diabetes_Mellitus=Diabetes_Mellitus
       self.Coronary_Artery_Disease=Coronary_Artery_Disease
       self.Appetite=Appetite
       self.Pedal_Edema=Pedal_Edema
       self.Anemia=Anemia

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict={'Age':[self.Age],'Blood Pressure':[self.Blood_Pressure],'Specific Gravity':[self.Specific_Gravity],
             'Albumin':[self.Albumin],'Sugar':[self.Sugar],'Red Blood Cells':[self.Red_Blood_Cells],
             'Pus Cell':[self.Pus_Cell],'Pus Cell clumps':[self.Pus_Cell_clumps],'Bacteria':[self.Bacteria],
             'Blood Glucose Random':[self.Blood_Glucose_Random],'Blood Urea':[self.Blood_Urea],
             'Serum Creatinine':[self.Serum_Creatinine],'Sodium':[self.Sodium],'Potassium':[self.Potassium]
             ,'Hemoglobin':[self.Hemoglobin],'Packed Cell Volume':[self.Packed_Cell_Volume],'White Blood Cell Count':[self.White_Blood_Cell_Count],
             'Red Blood Cell Count':[self.Red_Blood_Cell_Count],'Hypertension':[self.Hypertension],'Diabetes Mellitus':[self.Diabetes_Mellitus],
             'Coronary Artery Disease':[self.Coronary_Artery_Disease],'Appetite':[self.Appetite],'Pedal Edema':[self.Pedal_Edema],
             'Anemia':[self.Anemia]
             }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise(e,sys)
        








        