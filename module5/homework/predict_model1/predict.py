import pickle
from flask import Flask
from flask import request
from flask import jsonify

model_file="model1.bin"
with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)

app=Flask('churn_predict')

dv_file="dv.bin"
with open(dv_file, 'rb') as f_in:
    dv = pickle.load(f_in)

@app.route('/predict', methods=['POST'])
def predict():
	customer = request.get_json()
	X = dv.transform(customer)
	y_pred_proba=model.predict_proba(X)[0,1]
	churn = y_pred_proba >= 0.5

	result = {
		'churn_probability': y_pred_proba,
		'churn': bool(churn)
	}
	return jsonify(result)

if __name__ == "__main__":
	app.run(debug=True,  host='0.0.0.0', port=9696)
