from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
with open("src/thyroid_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# Load the scaler (if you used StandardScaler in Colab)
scaler = joblib.load("src/scaler.pkl")

# Define the features used in the model, updated with correct order and inclusion
selected_features = ['T3 Measured', 'T4U Measured', 'TSH Measured', 
                     'Tumor', 'TT4 Measured', 'Age', 'FTI', 
                     'T4U', 'T3', 'FTI Measured']  # Removed 'Psych' and added 'FTI Measured'

@app.route('/')
def index():
    return render_template('home.html')

@app.route("/moreinfo", methods=["GET", "POST"])
def moreinfo():
    return render_template('moreinfo.html')

@app.route("/predict", methods=["GET", "POST"])
def predict():
    return render_template('predict.html')

@app.route("/predictresult", methods=["GET", "POST"])
def predictresult():
    if request.method == "POST":
        try:
            # Collect form data with validation
            def get_float_value(field_name):
                value = request.form.get(field_name)
                if value is None or value == "":
                    raise ValueError(f"Field '{field_name}' is empty.")
                return float(value)

            # Attempt to collect and convert input data
            T3_Measured = get_float_value('T3_measured')
            T4U_Measured = get_float_value('T4U_measured')
            TSH_Measured = get_float_value('TSH_measured')
            Tumor = get_float_value('tumor')
            TT4_Measured = get_float_value('TT4_measured')
            Age = get_float_value('age')
            FTI = get_float_value('FTI')
            T4U = get_float_value('T4U')
            T3 = get_float_value('T3')
            FTI_Measured = get_float_value('FTI_measured')  # Changed to 'FTI Measured'

            # Create a NumPy array for the input
            input_data = np.array([[T3_Measured, T4U_Measured, TSH_Measured, Tumor,
                                    TT4_Measured, Age, FTI, T4U, T3, FTI_Measured]])

            # Create DataFrame and normalize the input data
            input_df = pd.DataFrame(input_data, columns=selected_features)
            input_df[selected_features] = scaler.transform(input_df[selected_features])

            # Make prediction
            pred = model.predict(input_df)

            # Map the prediction to the output label
            if pred == 0:
                res_Val = "Normal"
            elif pred == 1:
                res_Val = "Hyperthyroid"
            elif pred == 2:
                res_Val = "Hypothyroid"
            elif pred == 3:
                res_Val = "Goitre"  # Added condition for Goitre

            Output = f"Patient has {res_Val}"

            # Return the result to the prediction result page
            return render_template('predictresult.html', output=Output)

        except ValueError as ve:
            # Handle missing or invalid inputs with a custom error message
            return render_template('predictresult.html', output=str(ve))
        except Exception as e:
            # General error handling
            return render_template('predictresult.html', output="An unexpected error occurred. Please check your input values.")

    # If not POST, return to home page
    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=False)
