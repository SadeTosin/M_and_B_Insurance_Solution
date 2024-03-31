
import sys
sys.path.append(r'C:\Users\FOLASADE\OneDrive\Desktop\Projects\Insurance')
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:
            numerical_columns = ["age",
                "number_of_vehicles_involved",
                "witnesses",
                "vehicle_claim"]
            categorical_columns = [
                "gender",
                "incident_type",
                "collision_type",
                "incident_severity",
                "incident_city",
                "incident_hour_of_the_day",
                "police_report_available",
                "auto_make",
                "auto_model",
                "auto_year",
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    # def initiate_data_transformation(self, train_path, test_path):
    #     try:
    #         train_df = pd.read_csv(train_path)
    #         test_df = pd.read_csv(test_path)

    #         logging.info("Read train and test data completed")
    #         logging.info("Obtaining preprocessing object")

    #         preprocessing_obj = self.get_data_transformer_object()

    #         target_column_name = "fraud_reported"
    #         numerical_columns = ["age", "number_of_vehicles_involved", "witnesses", "vehicle_claim"]

    #         input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
    #         target_feature_train_df = train_df[target_column_name]
    #         print(input_feature_train_df)
    #         print(target_feature_train_df)

    #         input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
    #         target_feature_test_df = test_df[target_column_name]
    #         print(input_feature_test_df)
    #         print(target_feature_test_df)

    #         logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

    #         input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
    #         input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

    #         print("input_feature_train_arr shape:", input_feature_train_arr.shape)
    #         print("input_feature_test_arr shape:", input_feature_test_arr.shape)
    #         print("target_feature_train_df shape:", target_feature_train_df.shape)
    #         print("target_feature_test_df shape:", target_feature_test_df.shape)

    #         # Ensure input arrays have the same number of dimensions as target arrays
    #         # if input_feature_train_arr.ndim != target_feature_train_df.ndim:
    #         #     input_feature_train_arr = input_feature_train_arr.reshape(-1, 1)
    #         # if input_feature_test_arr.ndim != target_feature_test_df.ndim:
    #         #     input_feature_test_arr = input_feature_test_arr.reshape(-1, 1)

    #         # Ensure input arrays have the same number of dimensions as target arrays
    #         if input_feature_train_arr.ndim != target_feature_train_df.ndim:
    #             input_feature_train_arr = input_feature_train_arr.reshape(-1, 1) if input_feature_train_arr.ndim == 1 else input_feature_train_arr
    #         if input_feature_test_arr.ndim != target_feature_test_df.ndim:
    #             input_feature_test_arr = input_feature_test_arr.reshape(-1, 1) if input_feature_test_arr.ndim == 1 else input_feature_test_arr

    #         print("Shape of input_feature_train_arr:", input_feature_train_arr.shape)
    #         print("Shape of target_feature_train_df:", target_feature_train_df.shape)

    #         # Concatenate arrays
    #         # train_arr = np.hstack((input_feature_train_arr, target_feature_train_df))
    #         # test_arr = np.hstack((input_feature_test_arr, target_feature_test_df))

    #         # train_arr = np.hstack((input_feature_train_arr, target_feature_train_df.values.reshape(-1, 1)))
    #         # test_arr = np.hstack((input_feature_test_arr, target_feature_test_df.values.reshape(-1, 1)))

    #         train_arr = np.hstack((input_feature_train_arr, np.expand_dims(target_feature_train_df.values, axis=1)))
    #         test_arr = np.hstack((input_feature_test_arr, np.expand_dims(target_feature_test_df.values, axis=1)))




    #         print("Shape of train_arr after concatenation:", train_arr.shape)
    #         print("Shape of test_arr after concatenation:", test_arr.shape)

    #         logging.info("Saved preprocessing object.")
    #         save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
    #                     obj=preprocessing_obj)

    #         return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

    #     except Exception as e:
    #         raise CustomException(e, sys)
        







    # def initiate_data_transformation(self, train_path, test_path):
    #     try:
    #         train_df = pd.read_csv(train_path)
    #         test_df = pd.read_csv(test_path)

    #         logging.info("Read train and test data completed")
    #         logging.info("Obtaining preprocessing object")

    #         preprocessing_obj = self.get_data_transformer_object()

    #         target_column_name = "fraud_reported"
    #         numerical_columns = ["age", "number_of_vehicles_involved", "witnesses", "vehicle_claim"]

    #         input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
    #         target_feature_train_df = train_df[[target_column_name]]  # Keep target feature as DataFrame
    #         # print(input_feature_train_df)
    #         # print(target_feature_train_df)

    #         input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
    #         target_feature_test_df = test_df[[target_column_name]]  # Keep target feature as DataFrame
    #         print(input_feature_test_df)
    #         # print(target_feature_test_df)

    #         # print(target_feature_train_df.values)
    #         # print(target_feature_test_df.values)
    #         # print("Shape of target_feature_train_df.values:", target_feature_train_df.values.shape)
    #         # print("Shape of target_feature_test_df.values:", target_feature_test_df.values.shape)


    #         logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

    #         input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
    #         input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

    #         # print(input_feature_test_arr)
    #         print("input_feature_train_arr shape:", input_feature_train_arr.shape)
    #         print("input_feature_test_arr shape:", input_feature_test_arr.shape)
    #         print("target_feature_train_df shape:", target_feature_train_df.shape)
    #         print("target_feature_test_df shape:", target_feature_test_df.shape)

    #         logging.info("Saved preprocessing object.")
    #         save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
    #                     obj=preprocessing_obj)

    #         return input_feature_train_arr, target_feature_train_df.values, input_feature_test_arr, target_feature_test_df.values, self.data_transformation_config.preprocessor_obj_file_path

    #     except Exception as e:
    #         raise CustomException(e, sys)
        
    def initiate_data_transformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            logging.info("Obtaining preprocessing object")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name = "fraud_reported"
            numerical_columns = ["age", "number_of_vehicles_involved", "witnesses", "vehicle_claim"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
            )

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            # print(input_feature_train_df)
            # print(input_feature_train_arr)
            # print(target_feature_train_df)

            input_feature_train_arr_dense = input_feature_train_arr.toarray()
            input_feature_test_arr_dense = input_feature_test_arr.toarray()

            train_arr = np.c_[input_feature_train_arr_dense, np.array(target_feature_train_df)]
            #print(train_arr)

            test_arr = np.c_[input_feature_test_arr_dense, np.array(target_feature_test_df)]
            #print(train_arr)

            # train_arr = np.c_[
            #     input_feature_train_arr, np.array(target_feature_train_df)
            # ]
            #test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info(f"Saved preprocessing object.")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj

            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        except Exception as e:
            raise CustomException(e,sys)

            