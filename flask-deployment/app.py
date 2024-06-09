import pickle
import subprocess
from configparser import ConfigParser as cfgp
from flask import Flask, render_template, request, url_for

config = cfgp()
config.read('settings.cfg')
config_section = config['DEPLOYMENT']

model = pickle.load(open("model.pkl", "rb"))
port: int = int(config_section['PORT'])
app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    fields: list[str] = ['ROCE (%)', 'CASA (%)', 'Return on Equity/Networth (%)',
                         'Non-interest Income/Total Assets (%)', 'Operating Profit/Total Assets (%)', 'Operating Expenses/Total Assets (%)', 'Interest Expenses/Total Assets (%)', 'Face Value']

    return render_template("index.html", title="Inputs", subtitle="EPS Prediction using ML", fields=fields)


@app.route('/predict', methods=["POST"])
def predict():
    result: str = "Result : "

    v1: float = float(request.form.get('ROCE (%)', "0.00"))
    v2: float = float(request.form.get('CASA (%)', "0.00"))
    v3: float = float(request.form.get(
        'Return on Equity/Networth (%)', "0.00"))
    v4: float = float(request.form.get(
        'Non-interest Income/Total Assets (%)', "0.00"))
    v5: float = float(request.form.get(
        'Operating Profit/Total Assets (%)', "0.00"))
    v6: float = float(request.form.get(
        'Operating Expenses/Total Assets (%)', "0.00"))
    v7: float = float(request.form.get(
        'Interest Expenses/Total Assets (%)', "0.00"))
    v8: float = float(request.form.get('Face Value', "0.00"))

    try:
        pred = model.predict([[v1, v2, v3, v4, v5, v6, v7, v8]])
        result += str(pred[0])
    except:
        result += "Invalid input!"

    return render_template("result.html", result=result, title="Result")


if __name__ == "__main__":
    # subprocess.run(["npm", "i", "tailwindcss"], check=True)
    # subprocess.run(["npm", "run", "build:css"], check=True)
    app.run(debug=True, port=port)
