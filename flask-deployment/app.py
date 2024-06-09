import subprocess
from configparser import ConfigParser as cfgp
from flask import Flask, render_template, request, url_for

config = cfgp()
config.read('settings.cfg')
config_section = config['DEPLOYMENT']

port: int = int(config_section['PORT'])
app = Flask(__name__)


@app.route('/', methods=["GET"])
def index():
    fields: list[str] = ["ROCE (%)", "CASA (%)", "Return on Equity/Networth (%)",
                         "Non-interest Income/Total Assets (%)", "Operating Profit/Total Assets (%)", "Operating Expenses/Total Assets (%)", "Interest Expenses/Total Assets (%)", "Face Value"]
    return render_template("index.html", title="Inputs", subtitle="EPS Prediction using ML", fields=fields)


@app.route('/predict', methods=["POST"])
def predict():
    n1: str = request.form.get("ip1", "")
    n2: str = request.form.get("ip2", "")
    n3: str = request.form.get("ip3", "")
    n4: str = request.form.get("ip4", "")
    n5: str = request.form.get("ip5", "")
    n6: str = request.form.get("ip6", "")
    n7: str = request.form.get("ip7", "")
    n8: str = request.form.get("ip8", "")
    try:
        test: int = int(n1) + int(n2) + int(n3) + int(n4) + \
            int(n5) + int(n6) + int(n7) + int(n8)
        result: str = str(test)
    except:
        result: str = "Invalid input"
    return render_template("result.html", result=result, title="Result")


if __name__ == "__main__":
    # subprocess.run(["npm", "i", "tailwindcss"], check=True)
    # subprocess.run(["npm", "run", "build:css"], check=True)
    app.run(debug=True, port=port)
