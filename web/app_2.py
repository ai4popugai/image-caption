from flask import Flask, render_template, request
import os

from db import SQLiteDb
from dres_api.client import Client

NUM_KEY_FRAMES = 25

app = Flask(__name__)
client = Client()


WEB_DEFAULTS_FOLDER = os.path.join(os.getcwd(), 'web_defaults')
DB_PATH = os.path.join(WEB_DEFAULTS_FOLDER, 'v3c1')
DB = SQLiteDb(DB_PATH)


@app.route('/')
def index():
    return render_template('main_2.html')


@app.route('/exit', methods=['GET'])
def exit():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()

    os._exit(0)


if __name__ == '__main__':
    app.run(debug=False)