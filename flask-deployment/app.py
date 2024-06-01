import subprocess
from configparser import ConfigParser as cfgp
from flask import Flask, render_template, request, url_for

import re

config = cfgp()
config.read('settings.cfg')
config_section = config['DEPLOYMENT']

port: int = int(config_section['PORT'])
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == "POST":
        name = request.form.get("name")
        email = request.form.get("email")

        if name and email:
            name = name.strip()
            email = email.strip()

            name = re.sub(r'\d+', '', name)

        return render_template("result.html", name=name, email=email)

    return render_template("index.html", title="Credit Risk Modelling")


if __name__ == "__main__":
    # subrpocess.run(["npm", "i", "tailwindcss"], check=True)
    # subprocess.run(["npm", "run", "build:css"], check=True)
    app.run(debug=True, port=port)
