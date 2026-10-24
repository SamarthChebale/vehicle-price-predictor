from flask import Flask, request, render_template
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline
from src.pipeline.train_pipeline import TrainPipeline

application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train', methods=['GET', 'POST'])
def train_model():
    try:
        pipeline = TrainPipeline()
        r2_score = pipeline.run_pipeline()
        return f"Model trained successfully! Best Model R2 Score: {r2_score:.4f}"
    except Exception as e:
        return f"Error during model training: {str(e)}"

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        year = int(request.form.get('year'))
        km_driven = int(request.form.get('km_driven'))
        fuel = request.form.get('fuel')
        seller_type = request.form.get('seller_type')
        transmission = request.form.get('transmission')
        owner = request.form.get('owner')
        brand = request.form.get('brand', 'Maruti')
        car_model = request.form.get('car_model', '')

        if not car_model or not car_model.strip():
            car_model = f"{brand} Car"

        data = CustomData(
            year=year,
            km_driven=km_driven,
            fuel=fuel,
            seller_type=seller_type,
            transmission=transmission,
            owner=owner,
            brand=brand,
            car_model=car_model
        )

        pred_df = data.get_data_as_data_frame()
        print("Input DataFrame for Prediction:")
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        raw_price = max(0.0, float(results[0]))
        price_in_lakhs = round(raw_price / 100000.0, 2)
        raw_price_formatted = f"{int(raw_price):,}"

        return render_template(
            'index.html',
            results=price_in_lakhs,
            raw_price_formatted=raw_price_formatted,
            year=year,
            km_driven=km_driven,
            fuel=fuel,
            seller_type=seller_type,
            transmission=transmission,
            owner=owner,
            brand=brand,
            car_model=car_model
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
