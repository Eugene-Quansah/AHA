import json
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import logging

app = Flask(__name__)

# Setup rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per day", "50 per hour"]
)

# Load the saved model and scaler
try:
    model = joblib.load('chronic_kidney_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    logging.error(f"Model or scaler file not found: {e}")
    raise

# Load questions from the JSON file
try:
    with open('questions.json') as f:
        questions = json.load(f)
except FileNotFoundError as e:
    logging.error(f"Questions file not found: {e}")
    raise

# Create a SimpleImputer for handling missing values
imputer = SimpleImputer(strategy='mean')

@app.route('/')
def index():
    return render_template ('index.html')

@app.route('/questions')
@limiter.limit("10 per minute")
def show_questions():
    return render_template('questions.html', questions=questions)

@app.route('/predict', methods=['POST'])
@limiter.limit("5 per minute")
def predict():
    try:
        user_input = []
        skipped_questions = []
        for question in questions:
            answer = request.form.get(question["question"], None)
            if answer is None or answer == "":
                skipped_questions.append(question["question"])
                user_input.append(np.nan)  # Use NaN for skipped questions
            elif question["options"] is None:
                user_input.append(float(answer))  # For numeric questions
            else:
                # Map categorical answers to numbers
                answer_mapping = {
                    'normal': 1, 'abnormal': 0,
                    'present': 1, 'not present': 0,
                    'yes': 1, 'no': 0,
                    'good': 1, 'poor': 0
                }
                user_input.append(answer_mapping.get(answer.lower(), 0))

        # Convert user input to a NumPy array and reshape for prediction
        user_input = np.array(user_input).reshape(1, -1)

        # Impute missing values
        user_input_imputed = imputer.fit_transform(user_input)

        # Scale the input using the loaded scaler
        user_input_scaled = scaler.transform(user_input_imputed)

        # Predict using the loaded model
        prediction = model.predict(user_input_scaled)
        prediction_proba = model.predict_proba(user_input_scaled)[0]

        # Map the prediction to human-readable output
        result = "Chronic Kidney Disease (CKD) Detected" if prediction[0] == 1 else "No Chronic Kidney Disease (CKD) Detected"
        confidence = max(prediction_proba) * 100
        
        print(result)

        # Log the prediction
        logging.info(f"Prediction made: {result} with {confidence:.2f}% confidence")

        return render_template('result.html', 
                               result=result, 
                               confidence=confidence, 
                               skipped_questions=skipped_questions)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction. Please try again."}), 500

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded. Please try again later."}), 429

if __name__ == '__main__':
    app.run(debug=True)