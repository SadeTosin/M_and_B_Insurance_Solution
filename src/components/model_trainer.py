# import os
# import sys
# sys.path.append(r'C:\Users\FOLASADE\OneDrive\Desktop\Projects\Insurance')
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor
# from scipy.sparse import csr_matrix

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object,evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path=os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=ModelTrainerConfig()


#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             logging.info("Split training and test input data")

#             # Check if the target arrays are sparse matrices
#             if isinstance(train_array[:, -1], csr_matrix):
#                 y_train = train_array[:, -1].toarray().ravel()  # Convert to dense array and then ravel
#             else:
#                 y_train = train_array[:, -1].ravel()

#             if isinstance(test_array[:, -1], csr_matrix):
#                 y_test = test_array[:, -1].toarray().ravel()  # Convert to dense array and then ravel
#             else:
#                 y_test = test_array[:, -1].ravel()

#             X_train = train_array[:, :-1]
#             # y_train = train_array[:, -1].ravel()  # Convert to 1D array

#             X_test = test_array[:, :-1]
#             # y_test = test_array[:, -1].ravel()  # Convert to 1D array

#             print(y_test)
#             print("Shape of X_train:", X_train.shape)
#             print("Shape of y_train:", y_train.shape)
#             print("Shape of X_test:", X_test.shape)
#             print("Shape of y_test:", y_test.shape)

#             # Reshape 1D arrays to 2D if needed
#             # train_array = train_array.reshape(-1, 1) if train_array.ndim == 1 else train_array
#             # test_array = test_array.reshape(-1, 1) if test_array.ndim == 1 else test_array
#             # print("Shape of train_array:", train_array.shape)
#             # print("Shape of test_array:", test_array.shape)


#             # if train_array.ndim == 1:
#             #     train_array = train_array.reshape(-1, 1)
#             # if test_array.ndim == 1:
#             #     test_array = test_array.reshape(-1, 1)

#             #X_train, X_test, y_train, y_test = train_test_split(train_array[:, :-1], train_array[:, -1], test_size=0.2, random_state=42)
#             #X_train,y_train,X_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])

#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "Linear Regression": LinearRegression(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#                 "AdaBoost Regressor": AdaBoostRegressor(),
#             }
#             params={
#                 "Decision Tree": {
#                     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#                     # 'splitter':['best','random'],
#                     # 'max_features':['sqrt','log2'],
#                 },
#                 "Random Forest":{
#                     # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
#                     # 'max_features':['sqrt','log2',None],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "Gradient Boosting":{
#                     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
#                     'learning_rate':[.1,.01,.05,.001],
#                     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
#                     # 'criterion':['squared_error', 'friedman_mse'],
#                     # 'max_features':['auto','sqrt','log2'],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "Linear Regression":{},
#                 "XGBRegressor":{
#                     'learning_rate':[.1,.01,.05,.001],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "CatBoosting Regressor":{
#                     'depth': [6,8,10],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'iterations': [30, 50, 100]
#                 },
#                 "AdaBoost Regressor":{
#                     'learning_rate':[.1,.01,0.5,.001],
#                     # 'loss':['linear','square','exponential'],
#                     'n_estimators': [8,16,32,64,128,256]
#                 }
                
#             }

#             model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
#                                              models=models,param=params)
            
#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             logging.info(f"Best found model on both training and testing dataset")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted=best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square
            
#         except Exception as e:
#             raise CustomException(e,sys)
        

import os
import sys
sys.path.append(r'C:\Users\FOLASADE\OneDrive\Desktop\Projects\Insurance')
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.1:
                raise CustomException(e,sys)
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



            
        except Exception as e:
            raise CustomException(e,sys)
        