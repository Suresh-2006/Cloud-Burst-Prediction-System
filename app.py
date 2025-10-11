from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load model and scaler
model = joblib.load("cloudburst_model.pkl")
scaler = joblib.load("scaler.pkl")

# Home route
@app.route('/')
def home():
    return render_template('cloud_burst.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Read values from form
        temperature = float(request.form['temperature'])
        rainfall = float(request.form['rainfall'])
        evaporation = float(request.form['evaporation'])
        humidity = float(request.form['humidity'])
        wind = float(request.form['wind'])
        pressure = float(request.form['pressure'])

        # Create feature vector in same order as training
        features = np.array([[temperature, rainfall, evaporation, humidity, wind, pressure]])

        # Scale features
        scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(scaled)[0]

        # Return JSON result
        result = "Yes" if prediction == 1 else "No"
        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
