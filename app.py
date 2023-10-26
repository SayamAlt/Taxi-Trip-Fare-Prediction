from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

pipeline = joblib.load('pipeline.pkl')
transformer = pipeline.named_steps['transformer']
scaler = pipeline.named_steps['scaler']
model = pipeline.named_steps['model']

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict",methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        meter_rate = float(request.form['meter_rate'])
        trip_duration = float(request.form['trip_duration'])
        miscellaneous_fees = float(request.form['miscellaneous_fees'])
        tip = int(request.form['tip'])
        base_fare = float(request.form['base_fare'])
        data = pd.DataFrame([[meter_rate,trip_duration,miscellaneous_fees,tip,base_fare]],columns=['meter_rate','trip_duration','miscellaneous_fees','tip','base_fare'])
        transformed_data = transformer.transform(data)
        scaled_data = scaler.transform(transformed_data)
        pred = model.predict(scaled_data)
        output = '%.2f' % round(pred, 2)
        return render_template('index.html',prediction_text=f"The predicted total fare of your taxi trip is ₹{output}.")
    
if __name__ == '__main__':
    app.run(port=8000,debug=True)