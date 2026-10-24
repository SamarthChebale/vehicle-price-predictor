import os
import sys
import pickle
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException
from src.logger import logging

warnings.filterwarnings('ignore', category=ConvergenceWarning)

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Saved object successfully at path: {file_path}")

    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param=None):
    try:
        report = {}

        for model_name, model in models.items():
            para = param.get(model_name, {}) if param else {}

            if para:
                logging.info(f"Running GridSearchCV hyperparameter tuning for {model_name}...")
                gs = GridSearchCV(model, para, cv=3, n_jobs=-1, scoring='r2')
                gs.fit(X_train, y_train)

                model.set_params(**gs.best_params_)
                logging.info(f"Best parameters for {model_name}: {gs.best_params_}")
                print(f"[{model_name}] Best Parameters: {gs.best_params_}")

            logging.info(f"Training tuned model: {model_name}")
            model.fit(X_train, y_train)

            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            logging.info(f"Model: {model_name} | Train R2: {train_model_score:.4f} | Test R2: {test_model_score:.4f}")
            print(f"Model: {model_name:<23} | Train R2: {train_model_score:.4f} | Test R2: {test_model_score:.4f}")

            report[model_name] = test_model_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
