from flask import Flask, render_template, request
import joblib
import pandas as pd
import os




app = Flask(__name__)

# Load the trained machine learning model
model_filename = 'voting_classifier_model.pkl'
model = joblib.load(model_filename)

print('Model loaded successfully')


# Define the mapping for encoded values to meanings
prediction_mapping = {
    0: 'Not Suitable for Any',
    1: 'Only Cold Water Fish',
    2: 'Only Warm Water Fish',
    3: 'Warm & Cold Water Fish',
    4: 'Only Bacteria',
    5: 'Bacteria & Cold Water Fish',
    6: 'Bacteria & Warm Water Fish',
    7: 'Bacteria, Warm & Cold Water Fish',
    8: 'Only Plant',
    9: 'Plant & Cold Water Fish',
    10: 'Plant & Warm Water Fish',
    11: 'Plant, Warm & Cold Water Fish',
    12: 'Plant & Bacteria',
    13: 'Plant, Bacteria & Cold Water Fish',
    14: 'Plant, Bacteria & Warm Water Fish',
    15: 'Suitable for All'
}

# Define improvement suggestions for missing categories
improvement_tips = {
    "Bacteria": "Increase biological filtration or add nitrifying bacteria.",
    "Cold Water Fish": "Reduce temperature or improve oxygenation.",
    "Warm Water Fish": "Increase temperature slightly and maintain stable water conditions.",
    "Plant": "Ensure sufficient light and nutrient balance."
}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values
        ph = float(request.form['ph'])
        do = float(request.form['do'])
        temp = float(request.form['temp'])
        nh3 = float(request.form['nh3'])
        no2 = float(request.form['no2'])
        no3 = float(request.form['no3'])

        print(f'Inputs Received: pH={ph}, DO={do}, Temp={temp}, NH3={nh3}, NO2={no2}, NO3={no3}')

        # Create input DataFrame
        input_data = pd.DataFrame([[ph, do, temp, nh3, no2, no3]],
                                  columns=['pH', 'Dissolved Oxygen', 'Temperature', 'Ammonia', 'Nitrite', 'Nitrate'])

        print('Formatted Input Data:')
        print(input_data)

        # Get prediction
        prediction = model.predict(input_data)[0]
        print(f'Raw Model Prediction: {prediction}')

        # Get corresponding meaning
        prediction_result = prediction_mapping.get(prediction, 'Unknown Result')
        print(f'Prediction Result: {prediction_result}')

        # Generate improvement suggestions
        missing_categories = []
        if 'Plant' not in prediction_result:
            missing_categories.append("Plant")
        if 'Bacteria' not in prediction_result:
            missing_categories.append("Bacteria")
        if 'Cold Water Fish' not in prediction_result:
            missing_categories.append("Cold Water Fish")
        if 'Warm Water Fish' not in prediction_result:
            missing_categories.append("Warm Water Fish")

        # Create improvement text
        improvement_text = ""
        for category in missing_categories:
            improvement_text += f"To support {category}, {improvement_tips[category]}<br>"

        print(f'Improvement Suggestions: {improvement_text}')

        return render_template('index.html', prediction_text=f'Suitable for: {prediction_result}',
                               improvement_text=improvement_text)
    except Exception as e:
        print(f'Error occurred: {e}')
        return render_template('index.html', prediction_text='Error occurred during prediction!', improvement_text="")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # default 5000 for local
    app.run(host="0.0.0.0", port=port, debug=True)
