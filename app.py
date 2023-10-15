from flask import Flask, request, app, jsonify, render_template
import numpy as np
import pandas as pd
from joblib import dump, load
from src.utils import LoadClassifierThreshold

app = Flask(__name__)

# Load the model and preprocessor
model = LoadClassifierThreshold(model_path= 'notebook\model\\best_model.pkl',
                                threshold_path='notebook\model\\threshold.txt')
preprocessor = load('notebook\model\preprocessor.pkl')

# Create first route
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    # Create a DataFrame from the JSON data
    data_df = pd.DataFrame(data, index=[0])
    print(data_df)
    new_data = preprocessor.transform(data_df)
    output = model.predict_with_threshold(new_data)
    # Convert the output to an integer
    prediction = int(output[0])
    return jsonify(prediction)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the HTML form and create a DataFrame
    data = {
        'gender': [request.form['gender']],
        'age': [int(request.form['age'])],
        'hypertension': [int(request.form['hypertension'])],
        'heart_disease': [int(request.form['heart_disease'])],
        'ever_married': [request.form['ever_married']],
        'work_type': [request.form['work_type']],
        'Residence_type': [request.form['Residence_type']],
        'avg_glucose_level': [float(request.form['avg_glucose_level'])],
        'bmi': [float(request.form['bmi'])],
        'smoking_status': [request.form['smoking_status']]
    }
    data_df = pd.DataFrame(data)
    print(data_df)

    # Transform the data using the preprocessor
    final_input = preprocessor.transform(data_df)

    # Make predictions using the model
    output = model.predict_with_threshold(final_input)
    prediction = int(output[0])

    # # Define the prediction message
    # if prediction == 1:
    #     prediction_message = "There is an indication of stroke."
    # else:
    #     prediction_message = "There is no indication of stroke."

    return render_template("home.html", prediction_text=prediction)



if __name__ == '__main__':
    app.run(debug = True)