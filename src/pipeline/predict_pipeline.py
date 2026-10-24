import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            logging.info("Loading model and preprocessor objects...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            logging.info("Transforming input features...")
            data_scaled = preprocessor.transform(features)

            logging.info("Making prediction...")
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(
        self,
        year: int,
        km_driven: int,
        fuel: str,
        seller_type: str,
        transmission: str,
        owner: str,
        brand: str = "Maruti",
        car_model: str = None
    ):
        self.year = int(year)
        self.km_driven = int(km_driven)
        self.fuel = str(fuel)
        self.seller_type = str(seller_type)
        self.transmission = str(transmission)
        self.owner = str(owner)
        self.brand = str(brand)
        self.car_model = str(car_model) if car_model else f"{self.brand} Car"

        self.car_age = 2026 - self.year
        self.km_per_year = self.km_driven / (self.car_age + 1)

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "year": [self.year],
                "km_driven": [self.km_driven],
                "fuel": [self.fuel],
                "seller_type": [self.seller_type],
                "transmission": [self.transmission],
                "owner": [self.owner],
                "brand": [self.brand],
                "car_model": [self.car_model],
                "car_age": [self.car_age],
                "km_per_year": [self.km_per_year],
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info(f"Custom input data created:\n{df}")
            return df

        except Exception as e:
            raise CustomException(e, sys)
