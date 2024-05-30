""" Model deployment using Flask """

from flask import Flask, render_template
from configparser import ConfigParser as cfgp

config = cfgp()
config.read('settings.cfg')
config_section = config['DEPLOYMENT']

port: int = int(config_section['PORT'])
app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True, port=port)
