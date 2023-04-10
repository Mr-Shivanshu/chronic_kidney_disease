import pickle
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.logger import logging

application = Flask(__name__)
app = application

## Route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        logging.info('request is completed')
        return render_template('home.html')
    else:
        data = CustomData(
            Age=request.form.get('age'),
            Blood_Pressure=request.form.get('Blood_Pressure'),
            Specific_Gravity=request.form.get('Specific_Gravity'),
            Albumin=request.form.get('Albumin'),
            Sugar=request.form.get('Sugar'),
            Red_Blood_Cells=request.form.get('Red_Blood_Cells'),
            Pus_Cell=request.form.get('Pus_Cell'),
            Pus_Cell_clumps=request.form.get('Pus_Cell_clumps'),
            Bacteria=request.form.get('Bacteria'),
            Blood_Glucose_Random=request.form.get('Blood_Glucose_Random'),
            Blood_Urea=request.form.get('Blood_Urea'),
            Serum_Creatinine=request.form.get('Serum_Creatinine'),
            Sodium=request.form.get('Sodium'),
            Potassium=request.form.get('Potassium'),
            Hemoglobin=request.form.get('Hemoglobin'),
            Packed_Cell_Volume=request.form.get('Packed_Cell_Volume'),
            White_Blood_Cell_Count=request.form.get('White_Blood_Cell_Count'),
            Red_Blood_Cell_Count=request.form.get('Red_Blood_Cell_Count'),
            Hypertension=request.form.get('Hypertension'),
            Diabetes_Mellitus=request.form.get('Diabetes_Mellitus'),
            Coronary_Artery_Disease=request.form.get('Coronary_Artery_Disease'),
            Appetite=request.form.get('Appetite'),
            Pedal_Edema=request.form.get('Pedal_Edema'),
            Anemia=request.form.get('Anemia')
        )
        pred_df = data.get_data_as_dataframe() 
        print(pred_df)
        logging.info(f'pred_df completed{pred_df}')
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template('home.html', results=results[0])

if __name__ == '__main__':
    app.run(host="0.0.0.0",debug=True)
