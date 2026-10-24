import os
import sys
import warnings
from dataclasses import dataclass
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
    ExtraTreesRegressor,
    VotingRegressor
)
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.model_report = {}

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and testing input data...")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # Define base models
            m_cat = CatBoostRegressor(verbose=False, random_state=42, iterations=400, depth=6, learning_rate=0.08)
            m_xgb = XGBRegressor(random_state=42, n_estimators=200, max_depth=6, learning_rate=0.08)
            m_rf = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
            m_et = ExtraTreesRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)

            voting_ensemble = VotingRegressor(
                estimators=[
                    ('catboost', m_cat),
                    ('xgboost', m_xgb),
                    ('random_forest', m_rf),
                    ('extra_trees', m_et)
                ]
            )

            models = {
                "Voting Ensemble": voting_ensemble,
                "CatBoosting Regressor": m_cat,
                "XGBRegressor": m_xgb,
                "Random Forest": m_rf,
                "Extra Trees Regressor": m_et,
                "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                "Linear Regression": LinearRegression(),
                "Ridge": Ridge(random_state=42),
                "Lasso": Lasso(random_state=42),
                "Decision Tree": DecisionTreeRegressor(random_state=42),
                "AdaBoost Regressor": AdaBoostRegressor(random_state=42),
            }

            params = {
                "Voting Ensemble": {},
                "Decision Tree": {
                    'criterion': ['squared_error', 'friedman_mse', 'poisson'],
                    'max_depth': [5, 10, 15]
                },
                "Random Forest": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15]
                },
                "Extra Trees Regressor": {
                    'n_estimators': [100, 200],
                    'max_depth': [10, 15]
                },
                "Gradient Boosting": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [100, 200]
                },
                "Linear Regression": {},
                "Ridge": {
                    'alpha': [1.0, 10.0, 20.0, 50.0, 100.0, 200.0],
                    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
                },
                "Lasso": {
                    'alpha': [0.01, 0.1, 1.0, 10.0, 50.0, 100.0],
                    'max_iter': [1000, 2000, 3000],
                    'tol': [0.01, 0.001]
                },
                "XGBRegressor": {
                    'learning_rate': [0.08, 0.1],
                    'n_estimators': [100, 200]
                },
                "CatBoosting Regressor": {
                    'depth': [6, 8],
                    'learning_rate': [0.05, 0.08]
                },
                "AdaBoost Regressor": {
                    'learning_rate': [0.1, 0.05],
                    'n_estimators': [50, 100]
                }
            }

            logging.info("Evaluating models and ensemble architectures...")
            model_report: dict = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params
            )
            self.model_report = model_report

            print("\n" + "=" * 55)
            print(f"{'MODEL NAME':<30} | {'TEST R2 SCORE':<15}")
            print("-" * 55)
            for model_name, score in model_report.items():
                print(f"{model_name:<30} | {score:<15.4f}")
            print("=" * 55 + "\n")

            # Extract best model score
            best_model_score = max(sorted(model_report.values()))

            # Extract best model name
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found with R2 score greater than 0.6", sys)

            logging.info(f"Best model selected: {best_model_name} with R2 score: {best_model_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2_sq = r2_score(y_test, predicted)

            return r2_sq

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    from src.components.data_transformation import DataTransformation

    ingestion = DataIngestion()
    train_data, test_data = ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    model_trainer = ModelTrainer()
    r2_sq = model_trainer.initiate_model_trainer(train_arr, test_arr)
    print(f"Training completed successfully! Best Model R2 Score: {r2_sq:.4f}")

