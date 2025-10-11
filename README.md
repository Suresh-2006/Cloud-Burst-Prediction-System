# Cloudburst Prediction using Machine Learning (XGBoost)

## Overview

This project predicts the likelihood of a cloudburst or heavy rainfall using advanced machine learning techniques powered by XGBoost.
The model uses real meteorological data containing parameters like temperature, humidity, pressure, rainfall, and wind to identify patterns that indicate possible cloudburst conditions.

The system includes:

* A trained XGBoost model with tuned hyperparameters
* A Flask-based web interface for real-time prediction
* An intuitive frontend that displays clear results such as

  * High Chance of Cloudburst
  * Possible Heavy Rain
  * No Cloudburst Expected

This project was developed for the IBM Z Datathon 2025 by Team Datanauts (SAV241) from Saveetha Engineering College.

---

## Tech Stack

| Category             | Technology Used                                                   |
| -------------------- | ----------------------------------------------------------------- |
| Programming Language | Python 3                                                          |
| Libraries            | XGBoost, Scikit-learn, Pandas, Numpy, Seaborn, Matplotlib, Joblib |
| Web Framework        | Flask                                                             |
| Frontend             | HTML5, CSS3, JavaScript                                           |
| Deployment           | LinuxONE Cloud Server, IBM Cloud, or Localhost                    |
| Version Control      | Git and GitHub                                                    |

---

## Project Structure

```
Cloudburst_Prediction/
│
├── app.py                     # Flask backend for prediction
├── templates/
│   └── cloud_burst.html       # Frontend web page
│
├── cloudburst_model.pkl       # Trained ML model (XGBoost)
├── scaler.pkl                 # StandardScaler used for preprocessing
│
├── cloudburst_data.csv        # Dataset used for training and testing
├── cloud_burst.ipynb          # Jupyter notebook for model training
│
└── README.md                  # Project documentation
```

---

## Dataset Information

The dataset (cloudburst_data.csv) contains multiple meteorological observations and environmental factors used to train the model.
It includes the following key features:

| Feature                                     | Description                                |
| ------------------------------------------- | ------------------------------------------ |
| Date                                        | Date of observation                        |
| MinimumTemperature / MaximumTemperature     | Temperature readings                       |
| Rainfall / Evaporation / Sunshine           | Weather indicators                         |
| Humidity9am / Humidity3pm                   | Morning and afternoon humidity             |
| Pressure9am / Pressure3pm                   | Morning and afternoon atmospheric pressure |
| WindGustSpeed / WindSpeed9am / WindSpeed3pm | Wind details                               |
| Cloud9am / Cloud3pm                         | Cloud coverage                             |
| CloudBurstToday / CloudBurstTomorrow        | Target variable for prediction             |

---

## Model Training and Optimization

The machine learning pipeline includes several critical steps:

1. Data Preprocessing
   Missing values are filled with column means.
   The target variable CloudBurstTomorrow is encoded into binary classes (1 = Cloudburst, 0 = No Cloudburst).

2. Feature Engineering
   Derived new metrics like temperature range, humidity ratio, and pressure variation.
   Added seasonal and day-based encodings to capture temporal trends.

3. Feature Scaling
   StandardScaler was used to normalize continuous features for XGBoost compatibility.

4. Model Building (XGBoost)
   The XGBoost classifier was trained using a balanced scale_pos_weight to address class imbalance.
   Hyperparameters were tuned using GridSearchCV over 75 combinations and 3 folds, resulting in 225 total fits.
   Training was executed efficiently on a LinuxONE environment.

5. Performance Metrics
   Base Accuracy: 85.3 percent
   After threshold optimization: 90.2 percent
   Significant improvement observed in precision and recall for cloudburst event detection.

---

## Model Saving and Deployment

After training and evaluation, the best model and scaler were saved using:

```python
joblib.dump(best_model, "cloudburst_model.pkl")
joblib.dump(scaler, "scaler.pkl")
```

These files are later loaded by the Flask app for real-time predictions.

---

## Flask Web Application Overview

The web application enables users to input real-world weather parameters and instantly receive predictions.

How it works:

1. The user enters weather details.
2. The Flask backend loads the trained XGBoost model.
3. Input values are preprocessed and scaled.
4. The model outputs a cloudburst probability.
5. The result is displayed dynamically with a weather-themed interface.

---

## Running the Project Locally

1. Clone the Repository

   ```bash
   git clone https://github.com/<your-username>/Cloudburst-Prediction.git
   cd Cloudburst-Prediction
   ```

2. Install Dependencies

   ```bash
   pip install flask xgboost scikit-learn pandas matplotlib seaborn joblib
   ```

3. Run the Flask Application

   ```bash
   python app.py
   ```

4. Access in Browser

   ```
   http://127.0.0.1:5000/
   ```

---

## Deployment on LinuxONE or IBM Cloud

1. Upload the entire project folder to your LinuxONE virtual server.

2. Ensure all dependencies are installed:

   ```bash
   pip install flask xgboost scikit-learn pandas seaborn joblib
   ```

3. Run the Flask app in background:

   ```bash
   nohup python3 app.py &
   ```

4. Open the app in your browser using:

   ```
   http://<server-ip>:5000
   ```

---

## Results Summary

| Metric    | XGBoost Final |
| --------- | ------------- |
| Accuracy  | 90.2 percent  |
| Precision | 0.90          |
| Recall    | 0.64          |
| F1-Score  | 0.77          |

Confusion Matrix Summary:

* True Positive (TP): Correctly identified cloudburst days
* True Negative (TN): Correctly identified no-cloudburst days

---

## User Interface Preview

| Input Form                    | Prediction Output                                |
| ----------------------------- | ------------------------------------------------ |
| Example input form screenshot | Example output showing High Chance of Cloudburst |

---

## Contributors

Team Datanauts (SAV241)
Saveetha Engineering College
IBM Z Datathon 2025

| Name       | Role                                   |
| ---------- | -------------------------------------- |
| Your Name  | Model Development, Backend Integration |
| Teammate 1 | Frontend Design                        |
| Teammate 2 | Data Preprocessing                     |
| Teammate 3 | Deployment and Testing                 |

---

## Achievements

* Developed a complete end-to-end machine learning based cloudburst prediction system
* Achieved over 90 percent accuracy after optimization
* Successfully deployed on LinuxONE virtual environment
* Integrated interactive Flask web interface

---

## Future Enhancements

* Integrate real-time weather APIs for continuous data input
* Explore deep learning models like LSTM for time-series forecasting
* Build a mobile-compatible web UI for faster access

---

## License

This project is released under the MIT License.
You may use, modify, and distribute it for educational or research purposes.

---
