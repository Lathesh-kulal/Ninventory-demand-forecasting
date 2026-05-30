from flask import Flask, render_template, request
import joblib
import numpy as np
app = Flask(__name__)
# Load trained model
model = joblib.load('model.pkl')
@app.route('/')
def home():
    return render_template('index.html')
@app.route('/predict')
def predict_page():
    return render_template('predict.html')
@app.route('/result', methods=['POST'])
def result():
    store = int(request.form['store'])
    holiday = int(request.form['holiday'])
    temperature = float(request.form['temperature'])
    fuel_price = float(request.form['fuel_price'])
    cpi = float(request.form['cpi'])
    unemployment = float(request.form['unemployment'])
    day = int(request.form['day'])
    month = int(request.form['month'])
    year = int(request.form['year'])
    features = np.array([[
        store,
        holiday,
        temperature,
        fuel_price,
        cpi,
        unemployment,
        day,
        month,
        year
    ]])
    prediction = model.predict(features)
    predicted_sales = round(prediction[0], 2)
    return render_template('result.html', prediction=predicted_sales)
if __name__ == '__main__':
    app.run(debug=True)