from src.logger import logging
from src.exception import CustomException
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score
from src.utils import save_object,evaluate_models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
@dataclass

class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split array and test array")
            X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

            models={'RandomForest':RandomForestClassifier()}
                    #'Logistic_Regression':LogisticRegression()}
                    #'AdaBoost Regressor':AdaBoostClassifier(),
                    #'svc':SVC(),
                    #'DecisionTreeClassifier':DecisionTreeClassifier(),
                    #'GradientBoosting':GradientBoostingClassifier(),
                    #'XGBClassifier':XGBClassifier(),
                    #'CatBoostClassifier':CatBoostClassifier()
               

            params={
                
                'RandomForest' : {'n_estimators': [50, 100, 200],'max_depth': [5, 10, 20],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'max_features': ['sqrt', 'log2']}}
                #'Logistic_Regression' :{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty': ['l1', 'l2']}}
                #'AdaBoost Regressor': {'n_estimators': [50, 100, 200],'learning_rate': [0.1, 0.5, 1.0],'algorithm': ['SAMME', 'SAMME.R']},
                #'svc' : {'C': [0.1, 1, 10],'kernel': ['linear', 'poly', 'rbf'],'degree': [2, 3, 4],'gamma': ['scale', 'auto']},
                #'DecisionTreeClassifier':{'criterion':['gini','entropy',],'max_depth':[2,4,6,8,10], 'min_samples_split': [2, 4, 6, 8, 10],'min_samples_leaf': [1, 2, 3, 4, 5]},
                #'GradientBoosting' : {'n_estimators': [100, 500],'learning_rate': [0.1, 0.5],'max_depth': [3, 5]},
                #'XGBClassifier':{'learning_rate': [0.1, 0.01, 0.001],'max_depth': [3, 5, 7],'n_estimators': [50, 100, 200],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'gamma': [0, 0.1, 0.2],'reg_alpha': [0, 0.1, 0.5],'reg_lambda': [0, 0.1, 0.5]},
                #'CatBoostClassifier' : {'iterations': [100, 500, 1000],'learning_rate': [0.01, 0.05, 0.1],'depth': [3, 5, 7]} 
                 
            
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test, models=models,param=params)
            
             ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            

            
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(X_test)

            accuracy_score1=accuracy_score(y_test,predicted)

            return accuracy_score1,self.model_trainer_config.trained_model_file_path,best_model_name,best_model,best_model_score
        
        except Exception as e:
            logging.info(f"the error is {e}")
            raise CustomException(e,sys)
        

        
            

