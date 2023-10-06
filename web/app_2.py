import logging

import colorlog as colorlog
from flask import Flask, render_template, request, redirect, url_for
import os

from db import SQLiteDb
from dres_api.client import Client
from utils.web_utils import submit_match

app = Flask(__name__)
client = Client()


WEB_DEFAULTS_FOLDER = os.path.join(os.getcwd(), 'web_defaults')
DB_PATH = os.path.join(WEB_DEFAULTS_FOLDER, 'v3c1')
DB = SQLiteDb(DB_PATH)

# create logger
logger = logging.getLogger('submission logger')
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

# create colored formatter
formatter = colorlog.ColoredFormatter(
    '%(log_color)s%(levelname)s:%(name)s: %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'bold_red',
        'CRITICAL': 'red,bg_white'
    },
    secondary_log_colors={
        'message': {
            'ERROR': 'bold_red',
            'WARNING': 'yellow'
        }
    }
)

# set formatter for console handler
console_handler.setFormatter(formatter)


@app.route('/')
def index():
    return render_template('main_2.html')


@app.route('/render_main_page')
def render_main_page():
    return render_template('main_2.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        text = request.form.get('text')
        try:
            row_number = int(text)
        except ValueError:
            logger.error("Form input is not integer type. Nothing to submit")
            return redirect(url_for('render_main_page'))

        row = DB.get_row_by_id(row_number)
        if row is None:
            logger.error('Row is not exist in database. Nothing to submit')
            return redirect(url_for('render_main_page'))

        submit_match(client, row)
        logger.info('Solution was sent')

        return redirect(url_for('render_main_page'))


@app.route('/exit', methods=['GET'])
def exit():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()

    os._exit(0)


if __name__ == '__main__':
    app.run(debug=False)