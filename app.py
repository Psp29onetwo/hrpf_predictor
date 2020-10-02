from flask import Flask, request, jsonify, render_template, url_for
from flask_restful import Api, Resource
import model



app = Flask(__name__, template_folder='template')
api = Api(app)
#'time','ejection_fraction','serum_creatinine','age'

@app.route("/")
def homepage():
    return render_template("form.html")


@app.route("/predict", methods=['POST'])
def predict():
    if request.method == "POST":
        time = float(request.form['time'])
        ejectionFraction = float(request.form['ej_fraction'])
        serum_Creatinine = float(request.form['serum_creatinine'])
        age = float(request.form['age'])
   
    features = [time, ejectionFraction, serum_Creatinine, age]
    result = model.predict(features)
    prediction = int(result['live'])
    data = ''
    if prediction == 1:
        data = 'This heart failure can be fatal.'
    else:
        data = 'There is no chance of fatality.'
    return render_template("index.html", data = data)
    


if __name__ == "__main__":
    app.run(debug=True)