import pickle
import subprocess
from flask import Flask, render_template, request, url_for

result: str = ""
values: list[float] = []
subtitle: str = "EPS Prediction using ML"
fields: list[str] = ['ROCE (%)', 'CASA (%)', 'Return on Equity/Networth (%)',
                     'Non-interest Income/Total Assets (%)', 'Operating Profit/Total Assets (%)', 'Operating Expenses/Total Assets (%)', 'Interest Expenses/Total Assets (%)', 'Face Value']

model = pickle.load(open("model.pkl", "rb"))
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    result: str = ""
    values: list[float] = []
    if request.method == "POST":
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

        values = [v1, v2, v3, v4, v5, v6, v7, v8]
        try:
            pred = model.predict([values])
            result += str(pred[0])
        except:
            result += ""
        return render_template("index.html", title="Result", subtitle=subtitle, fields=fields, values=values, result=result)

    return render_template("index.html", title="Inputs", subtitle=subtitle, fields=fields, values=values, result=result)


if __name__ == "__main__":
    subprocess.run(["npm", "i", "tailwindcss"], check=True)
    subprocess.run(["npm", "run", "build:css"], check=True)

    app.run(debug=True)
