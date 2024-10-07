from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the scaler and model
scaler = StandardScaler()
model = joblib.load('model.pkl')

app = Flask(__name__)
app = Flask(__name__, static_url_path='/static')
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    mileage = float(request.form['mileage'])
    enginev = float(request.form['enginev'])
    brand_bmw = int(request.form['BMW'])
    brand_mercedes = int(request.form['Mercedes-Benz'])
    brand_mitsubishi = int(request.form['Mitsubishi'])
    brand_renault = int(request.form['Renault'])
    brand_toyota = int(request.form['Toyota'])
    brand_volkswagen = int(request.form['Volkswagen'])
    body_hatch = int(request.form['hatch'])
    body_other = int(request.form['other'])
    body_sedan = int(request.form['sedan'])
    body_vagon = int(request.form['vagon'])
    body_van = int(request.form['van'])
    engine_type_gas = int(request.form['Gas'])
    engine_type_other = int(request.form['Other'])
    engine_type_petrol = int(request.form['Petrol'])
    registration_yes = int(request.form['registration'])

    # Create a DataFrame with the input features
    input_data = pd.DataFrame({
        'mileage': [mileage],
        'enginev': [enginev],
        'brand_bmw': [brand_bmw],
        'brand_mercedes': [brand_mercedes],
        'brand_mitsubishi': [brand_mitsubishi],
        'brand_renault': [brand_renault],
        'brand_toyota': [brand_toyota],
        'brand_volkswagen': [brand_volkswagen],
        'body_hatch': [body_hatch],
        'body_other': [body_other],
        'body_sedan': [body_sedan],
        'body_vagon': [body_vagon],
        'body_van': [body_van],
        'engine_type_gas': [engine_type_gas],
        'engine_type_other': [engine_type_other],
        'engine_type_petrol': [engine_type_petrol],
        'registration_yes': [registration_yes]
    })

    # Scale the input data
    input_scaled = scaler.fit_transform(input_data.to_numpy())

    # Make predictions
    y_pred = model.predict(input_scaled)

    # Print the final result
    result = np.exp(y_pred)
    print(result)

    return render_template('index.html', price=result)

if __name__ == '__main__':
    app.run(debug=True)
