from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd
import sqlite3

app = Flask(__name__)

# Load the saved model
with open('models/sleep_disorder_model.pkl', 'rb') as model_file:
    saved_data = pickle.load(model_file)
    model = saved_data['model']
    scaler = saved_data['scaler']
    le_gender = saved_data['le_gender']
    le_occupation = saved_data['le_occupation']
    le_bmi = saved_data['le_bmi']
    le_sleep_disorder = saved_data['le_sleep_disorder']

@app.route('/')
def home():
    # Get unique categories for dropdowns
    genders = le_gender.classes_
    occupations = le_occupation.classes_
    bmi_categories = le_bmi.classes_

    return render_template('index.html', 
                           genders=genders, 
                           occupations=occupations, 
                           bmi_categories=bmi_categories)


@app.route('/dashboard')
def dashboard():
    # Connect to the SQLite database
    conn = sqlite3.connect('sleep_predictions.db')
    cursor = conn.cursor()

    # Fetch the most recent prediction
    recent_prediction = cursor.execute('''
        SELECT * FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT 1
    ''').fetchone()

    # Fetch historical readings (last 3 entries)
    historical_readings = cursor.execute('''
        SELECT timestamp, sleep_duration, quality_of_sleep, 
               predicted_sleep_disorder, 
               ROUND(RANDOM() * 100, 2) as confidence 
        FROM predictions 
        ORDER BY timestamp DESC 
        LIMIT 3
    ''').fetchall()

    # Close the database connection
    conn.close()

    # Prepare the data for the template
    if recent_prediction and historical_readings:
        dashboard_data = {
            'recent_prediction': {
                'disorder': recent_prediction[12] or 'No Disorder',
                'confidence': round(len(historical_readings) * 25, 2),  # Simple confidence calculation
                'risk_level': 'Moderate'
            },
            'health_metrics': {
                'age': recent_prediction[2],
                'gender': recent_prediction[3],
                'sleep_duration': recent_prediction[5],
                'sleep_quality': recent_prediction[6],
                'stress_level': recent_prediction[7],
                'physical_activity': recent_prediction[6],
                'heart_rate': recent_prediction[10],
                'daily_steps': recent_prediction[11],
                'blood_pressure': f"{recent_prediction[9]}/80"  # Assuming diastolic is 80
            },
            'historical_readings': [
                {
                    'date': reading[0],
                    'sleep_duration': reading[1],
                    'sleep_quality': 'High' if reading[2] > 7 else 'Moderate' if reading[2] > 5 else 'Low',
                    'prediction': reading[3],
                    'confidence': reading[4]
                } for reading in historical_readings
            ]
        }
        return render_template('dashboard.html', data=dashboard_data)
    
    # Fallback if no data
    return render_template('dashboard.html', data=None)


@app.route('/form')
def forms():
    return render_template('form.html')

@app.route('/camera')
def camera():
    return render_template('camera.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Collect form data
    gender = request.form['gender']
    occupation = request.form['occupation']
    bmi_category = request.form['bmi_category']

    # Encode categorical variables
    gender_encoded = le_gender.transform([gender])[0]
    occupation_encoded = le_occupation.transform([occupation])[0]
    bmi_encoded = le_bmi.transform([bmi_category])[0]

    # Extract systolic BP
    systolic_bp = int(request.form['blood_pressure'].split('/')[0])

    # Collect features for prediction
    features = [
        float(request.form['age']),
        gender_encoded,
        occupation_encoded,
        float(request.form['sleep_duration']),
        float(request.form['quality_of_sleep']),
        float(request.form['physical_activity_level']),
        float(request.form['stress_level']),
        bmi_encoded,
        systolic_bp,
        float(request.form['heart_rate']),
        float(request.form['daily_steps'])
    ]
    
    # Scale the features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)
    prediction_label = le_sleep_disorder.inverse_transform(prediction)[0]
    
    # Connect to SQLite database
    try:
        # Establish database connection
        conn = sqlite3.connect('sleep_predictions.db')
        cursor = conn.cursor()
        
        # Create table if not exists
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                age REAL,
                gender TEXT,
                occupation TEXT,
                sleep_duration REAL,
                quality_of_sleep REAL,
                physical_activity_level REAL,
                stress_level REAL,
                bmi_category TEXT,
                systolic_bp INTEGER,
                heart_rate REAL,
                daily_steps REAL,
                predicted_sleep_disorder TEXT
            )
        ''')
        
        # Insert prediction data
        cursor.execute('''
            INSERT INTO predictions (
                age, gender, occupation, sleep_duration, quality_of_sleep, 
                physical_activity_level, stress_level, bmi_category, 
                systolic_bp, heart_rate, daily_steps, predicted_sleep_disorder
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            float(request.form['age']),
            gender,
            occupation,
            float(request.form['sleep_duration']),
            float(request.form['quality_of_sleep']),
            float(request.form['physical_activity_level']),
            float(request.form['stress_level']),
            bmi_category,
            systolic_bp,
            float(request.form['heart_rate']),
            float(request.form['daily_steps']),
            prediction_label
        ))
        
        # Commit changes and close connection
        conn.commit()
    except sqlite3.Error as e:
        # Log the error (you might want to use proper logging in a production app)
        print(f"Database error: {e}")
    finally:
        # Ensure connection is closed
        if conn:
            conn.close()
    
    return render_template('result.html', prediction=prediction_label)

if __name__ == '__main__':
    app.run(debug=True)