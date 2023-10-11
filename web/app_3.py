import logging

import colorlog as colorlog
from flask import Flask, render_template, request, redirect, url_for
import os

from db import SQLiteDb
from dres_api.client import Client
from utils.web_utils import submit_match

app = Flask(__name__)
client = Client()

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
    return render_template('main_3.html')


@app.route('/render_main_page')
def render_main_page():
    return render_template('main_3.html')


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        video_id = request.form.get('video_id')
        frame_number = request.form.get('frame_number')
        try:
            video_id = int(video_id)
            frame_number = int(frame_number)
        except ValueError:
            logger.error("Form inputs is not integer type. Nothing to submit")
            return redirect(url_for('render_main_page'))

        client.submit(item='%05d' % video_id, frame=frame_number)
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