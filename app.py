
# app.py
from flask import Flask, render_template, request, redirect, url_for
from flask import session
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load the trained model
loaded_model_d = pickle.load(open(r'C:\Users\Kanika Rana\PycharmProjects\flaskProject\venv\trained_model_d.sav', 'rb'))
loaded_model_h = pickle.load(open(r'C:\Users\Kanika Rana\PycharmProjects\flaskProject\venv\trained_model_h.sav', 'rb'))
loaded_model_p = pickle.load(open(r'C:\Users\Kanika Rana\PycharmProjects\flaskProject\venv\trained_model_p.sav', 'rb'))
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model_d.predict(input_data_reshaped)
    return prediction[0]

def heart_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model_h.predict(input_data_reshaped)
    return prediction[0]

def parkinson_prediction(input_data):
    input_data_as_numpy_array=np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = loaded_model_p.predict(input_data_reshaped)
    return prediction[0]
app.secret_key = 'your_secret_key'

# Disease detection route
# Profile route
@app.route('/profile/<username>', methods=['GET', 'POST'])
def profile(username):
    if 'logged_in' not in session or 'username' not in session or session['username'] != username:
        return redirect(url_for('login'))

    if request.method == 'POST':
        try:
            # Assuming you have a form with relevant input fields for disease detection
            input_data = [
                float(request.form['Feature1']),
                float(request.form['Feature2']),
                # Add more features as needed
            ]

            # Perform disease detection based on the selected disease type
            disease_type = request.form['disease_type']

            if disease_type == 'diabetes':
                prediction = diabetes_prediction(input_data)
                diagnosis = 'The person is diabetic' if prediction == 1 else 'The person is not diabetic'
            elif disease_type == 'heart':
                prediction = heart_prediction(input_data)
                diagnosis = 'The person has Heart Disease' if prediction == 1 else 'The person does not have Heart Disease'
            elif disease_type == 'parkinson':
                prediction = parkinson_prediction(input_data)
                diagnosis = 'The person has Parkinson\'s Disease' if prediction == 1 else 'The person does not have Parkinson\'s Disease'
            else:
                diagnosis = 'Invalid disease type selected'

            return render_template('result.html', diagnosis=diagnosis)

        except ValueError:
            error_message = "Invalid input. Please enter valid numerical values."
            return render_template('app1.html', error_message=error_message)

    return render_template('profile.html', username=username)

# Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        # Perform login authentication here
        # Assuming a successful login for demonstration purposes
        session['logged_in'] = True
        session['username'] = request.form['username']

        # Use the correct URL for the 'profile' endpoint
        return redirect(url_for('profile', username=session['username']))

    return render_template('login.html')

# Logout route
@app.route('/logout')
def logout():
    # Clear the session data
    session.pop('logged_in', None)
    session.pop('username', None)
    return redirect(url_for('login'))

# Signup route
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        # Get user input from the form
        username = request.form['username']
        password = request.form['password']

        # You might want to add more validation and security checks here

        # For demonstration purposes, let's just store the username in the session
        session['username'] = username

        # Redirect to the user's profile (you can change this based on your application flow)
        return redirect(url_for('profile', username=username))

    # If it's a GET request, render the signup form
    return render_template('signup.html')  # Create a signup.html template for your signup form

@app.route('/')
def index():
    return render_template('app1.html')

@app.route('/location')
def location():
    return render_template('location.html')

@app.route('/bookapp')
def bookapp():
    return render_template('bookapp.html')

@app.route('/aboutDoctors')
def aboutDoctors():
    return render_template('aboutDoctors.html')

@app.route('/reviews')
def reviews():
    return render_template('reviews.html')

@app.route('/add_reviews')
def add_reviews():
    return render_template('add_reviews.html')
@app.route('/index2')
def index2():
    return render_template('index2.html')

@app.route('/indexheart')
def indexheart():
    return render_template('indexheart.html')

@app.route('/indexpark')
def indexpark():
    return render_template('indexpark.html')
@app.route('/contactUs')
def contactUs():
    return render_template('contactUs.html')

@app.route('/news')
def news():
    return render_template('news.html')

@app.route('/Insurance')
def Insurance():
    return render_template('Insurance.html')

@app.route('/aboutUs')
def aboutUs():
    return render_template('aboutUs.html')

@app.route('/result', methods=['POST'])
def result():
    if request.method == 'POST':
        try:
            pregnancies = float(request.form['Pregnancies'])
            glucose = float(request.form['Glucose'])
            bloodpressure = float(request.form['BloodPressure'])
            skinthickness = float(request.form['SkinThickness'])
            insulin = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            diabetespedigreefunction = float(request.form['DiabetesPedigreeFunction'])
            age = float(request.form['Age'])

            input_data = [pregnancies, glucose, bloodpressure, skinthickness, insulin, bmi, diabetespedigreefunction, age]
            prediction = diabetes_prediction(input_data)

            if prediction == 0:
                diagnosis = 'The person is not diabetic'
            else:
                diagnosis = 'The person is diabetic'

            return render_template('result.html', diagnosis=diagnosis)

        except ValueError:
            error_message = "Invalid input. Please enter valid numerical values."
            return render_template('app1.html', error_message=error_message)

@app.route('/resultheart', methods=['POST'])
def resultheart():
    if request.method == 'POST':
        try:
            age = float(request.form['Age'])
            sex = float(request.form['Sex'])
            cp = float(request.form['ChestPainType'])
            trestbps = float(request.form['RestingBloodPressure'])
            chol = float(request.form['Cholesterol'])
            fbs = float(request.form['FastingBloodSugar'])
            restecg = float(request.form['RestingECG'])
            thalach = float(request.form['MaxHeartRate'])
            exang = float(request.form['ExerciseAngina'])
            oldpeak = float(request.form['Oldpeak'])
            slope = float(request.form['Slope'])
            ca = float(request.form['NumVesselsColored'])
            thal = float(request.form['Thalassemia'])

            input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
            prediction = heart_prediction(input_data)

            if prediction == 0:
                diagnosis = 'The person does not have Heart Disease'
            else:
                diagnosis = 'The person has Heart Disease'

            return render_template('result.html', diagnosis=diagnosis)

        except ValueError:
            error_message = "Invalid input. Please enter valid numerical values."
            return render_template('app1.html', error_message=error_message)

@app.route('/resultpark', methods=['POST'])
def resultpark():
    if request.method == 'POST':
        try:
            # Extracting values from the form
            mdvp_fo = float(request.form['MDVPFo'])
            mdvp_fhi = float(request.form['MDVPFhi'])
            mdvp_flo = float(request.form['MDVPFlo'])
            mdvp_jitter = float(request.form['MDVPJitter'])
            mdvp_jitter_abs = float(request.form['MDVPJitterAbs'])
            mdvp_rap = float(request.form['MDVPRAP'])
            mdvp_ppq = float(request.form['MDVPPPQ'])
            mdvp_jitter_ddp = float(request.form['MDVPJitterDDP'])
            mdvp_shimmer = float(request.form['MDVPShimmer'])
            mdvp_shimmer_db = float(request.form['MDVPShimmerdB'])
            mdvp_shimmer_apq3 = float(request.form['MDVPShimmerAPQ3'])
            mdvp_shimmer_apq5 = float(request.form['MDVPShimmerAPQ5'])
            mdvp_apq = float(request.form['MDVPAPQ'])
            mdvp_shimmer_dda = float(request.form['MDVPShimmerDDA'])
            nhr = float(request.form['NHR'])
            hnr = float(request.form['HNR'])
            rpde = float(request.form['RPDE'])
            dfa = float(request.form['DFA'])
            spread1 = float(request.form['spread1'])
            spread2 = float(request.form['spread2'])
            d2 = float(request.form['D2'])
            ppe = float(request.form['PPE'])

            # Creating input data array
            input_data = [mdvp_fo, mdvp_fhi, mdvp_flo, mdvp_jitter, mdvp_jitter_abs, mdvp_rap, mdvp_ppq,
                          mdvp_jitter_ddp, mdvp_shimmer, mdvp_shimmer_db, mdvp_shimmer_apq3, mdvp_shimmer_apq5, mdvp_apq, mdvp_shimmer_dda,
                          nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]

            # Calling the Parkinson prediction function
            prediction = parkinson_prediction(input_data)

            # Checking the prediction and creating diagnosis
            if prediction == 0:
                diagnosis = 'The person does not have Parkinson\'s Disease'
            else:
                diagnosis = 'The person has Parkinson\'s Disease'

            # Rendering the result template with the diagnosis
            return render_template('resultpark.html', diagnosis=diagnosis)

        except ValueError:
            error_message = "Invalid input. Please enter valid numerical values."
            return render_template('app_parkinson.html', error_message=error_message)

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=8080)