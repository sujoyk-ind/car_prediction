from flask import Flask,render_template,request
from flask_cors import CORS,cross_origin
import pandas as pd
import numpy as np
import pickle

app = Flask(__name__,template_folder='.')

cors=CORS(app)
model = pickle.load(open('LinearRegressionModel.pkl','rb'))
car = pd.read_csv('cleaned_car_data.csv')

@app.route('/',methods=['GET','POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = car.groupby('company')['name'].apply(list).to_dict()
    year = sorted(car['year'].unique(),reverse=True)
    fuel_type = car['fuel_type'].unique()

    companies.insert(0,'Select Company')
    year.insert(0,'Select Year')
    # fuel_type.insert(0,'Select Fuel Type')

    return render_template('index.html',companies = companies,car_models = car_models, years = year, fuel_type= fuel_type)

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    company=request.form.get('company')

    car_model=request.form.get('car-model')
    year=request.form.get('years')
    fuel_type=request.form.get('fuel_type')
    kms_driven=request.form.get('kilo-driven')


    # Input validation: Ensure all fields are filled
    if company == 'Select Company' or not company:
        return "Please select a valid company."
    if car_model == 'Select Model' or not car_model:
        return "Please select a valid car model."
    if not year or year == 'Select Year':
        return "Please select a valid year."
    if not fuel_type:
        return "Please select a valid fuel type."
    if not kms_driven or not kms_driven.isdigit():
        return "Please enter valid kilometers driven."

    # Check if any value is missing or empty before passing to model
    input_data = np.array([car_model, company, year, kms_driven, fuel_type]).reshape(1, 5)
    if any(val is None or val == '' for val in input_data[0]):
        return "One or more input fields are missing or invalid. Please check your inputs."

    try:
        # Make prediction
        prediction = model.predict(pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                                                 data=input_data))
        return str(np.round(prediction[0], 2))
    
    except Exception as e:
        return f"An error occurred during prediction: {str(e)}"


if __name__ == "__main__":
    app.run(debug=True)