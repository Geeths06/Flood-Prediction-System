from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

# Initialize Flask App
app = Flask(__name__)

# Load Model and Scaler
model = joblib.load("flood_model.pkl")
scaler = joblib.load("scaler.pkl")


# Home Page
@app.route('/')
def home():
    return render_template('home.html')


# Prediction Page
@app.route('/Predict')
def predict_page():
    return render_template('index.html')


# Prediction Logic
@app.route('/predict', methods=['POST'])
def predict():

    try:

        temperature = float(request.form['Temperature'])
        humidity = float(request.form['Humidity'])
        cloud_cover = float(request.form['Cloud Cover'])

        data = pd.DataFrame(
            [[
                temperature,
                humidity,
                cloud_cover
            ]],
            columns=[
                "Temperature",
                "Humidity",
                "Cloud Cover"
            ]
        )

        scaled_data = scaler.transform(data)

        prediction = model.predict(scaled_data)

        if prediction[0] == 1:
            return redirect(url_for("chance"))
        else:
            return redirect(url_for("no_chance"))

    except Exception as e:
        return f"Error : {e}"


# Flood Predicted Page
@app.route('/chance')
def chance():
    return render_template("chance.html")


# No Flood Page
@app.route('/no_chance')
def no_chance():
    return render_template("no_chance.html")


# Run Application
if __name__ == "__main__":
    app.run(debug=True)