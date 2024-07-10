from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route("/")
def introduction():
    return render_template("introduction.html")

@app.route("/predict")
def show_form():
    return render_template("predict.html")

@app.route('/predict', methods=['POST'])
def predict_fraud():
    # Extract form data
    step = int(request.form.get('Step'))
    Type = int(request.form.get('Type'))
    Amount = float(request.form.get('Amount'))
    Name_of_the_Originator = request.form.get('Name_of_the_Originator')
    oldbalanceOrg = float(request.form.get('oldbalanceOrg'))
    newbalanceOrig = float(request.form.get('newbalanceOrig'))
    Name_of_the_Reciever = request.form.get('Name_of_the_Reciever')
    oldbalanceDest = float(request.form.get('oldbalanceDest'))
    newbalanceDest = float(request.form.get('newbalanceDest'))

    # Perform prediction
    input_data = np.array([step, Type, Amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest]).reshape(1, -1)
    result = model.predict(input_data)

    # Interpret prediction result
    if result.all() == 0:
        prediction = 'Not Fraud'
    else:
        prediction = 'Fraud'

    return render_template("result.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
