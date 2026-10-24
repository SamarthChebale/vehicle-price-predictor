import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Creates and returns the ColumnTransformer pipeline object
        '''
        try:
            numerical_columns = ['year', 'km_driven', 'car_age', 'km_per_year']
            categorical_columns = ['fuel', 'seller_type', 'transmission', 'owner', 'brand', 'car_model']

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("one_hot_encoder", OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed.")

            # Advanced Feature Engineering
            for df in [train_df, test_df]:
                if 'name' in df.columns:
                    df['brand'] = df['name'].apply(lambda x: str(x).split()[0])
                    df['car_model'] = df['name'].apply(lambda x: ' '.join(str(x).split()[:2]))
                else:
                    if 'brand' not in df.columns:
                        df['brand'] = 'Maruti'
                    if 'car_model' not in df.columns:
                        df['car_model'] = df['brand'] + ' Car'

                df['car_age'] = 2026 - df['year']
                df['km_per_year'] = df['km_driven'] / (df['car_age'] + 1)

            logging.info("Obtaining preprocessing object...")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "selling_price"
            drop_columns = [target_column_name, "name"]

            input_feature_train_df = train_df.drop(columns=[col for col in drop_columns if col in train_df.columns])
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[col for col in drop_columns if col in test_df.columns])
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes...")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saving preprocessing object...")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)
