import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        gender: str,
        incident_type: str,
        collision_type: str,
        incident_severity: str,
        incident_city: str,
        incident_hour_of_the_day: str,	
        police_report_available: str,
        auto_make: str,
        auto_model: str,
        auto_year: str,
        age: int,
        number_of_vehicles_involved: int,
        witnesses: int,
        vehicle_claim: int):

        self.gender = gender

        self.incident_type = incident_type

        self.collision_type = collision_type

        self.incident_severity = incident_severity

        self.incident_city = incident_city

        self.incident_hour_of_the_day = incident_hour_of_the_day

        self.police_report_available = police_report_available

        self.auto_make = auto_make

        self.auto_model = auto_model

        self.auto_year = auto_year

        self.age = age

        self.number_of_vehicles_involved = number_of_vehicles_involved

        self.witnesses = witnesses

        self.vehicle_claim = vehicle_claim

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "incident_type": [self.incident_type],
                "collision_type": [self.collision_type],
                "incident_severity": [self.incident_severity],
                "incident_city": [self.incident_city],
                "incident_hour_of_the_day": [self.incident_hour_of_the_day],
                "police_report_available": [self.police_report_available],
                "auto_make": [self.auto_make],
                "auto_model": [self.auto_model],
                "auto_year": [self.auto_year],
                "age": [self.age],
                "number_of_vehicles_involved": [self.number_of_vehicles_involved],
                "witnesses": [self.witnesses],
                "vehicle_claim": [self.vehicle_claim]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)

