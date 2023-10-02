import random
import shutil
from typing import List

from flask import Flask, request, render_template, redirect, url_for
import os

from datasets.classification.gpr import GPRDataset
from db import SQLiteDb
from dres_api.client import Client
from nn_models.object_detection.yolov7.yolo_inference import YOLO_NAMES
from web.preprocess_video import preprocess_videos

NUM_KEY_FRAMES = 25

app = Flask(__name__)
client = Client()

WEB_DEFAULTS_FOLDER = os.path.join(os.getcwd(), 'web_defaults')
DB_DEFAULT_PATH = os.path.join(WEB_DEFAULTS_FOLDER, 'v3c1')
DB_DEFAULT = SQLiteDb(DB_DEFAULT_PATH)

WEB_EXECUTION_FOLDER = os.path.join(os.getcwd(), 'web_runtime')

UPLOAD_FOLDER = os.path.join(WEB_EXECUTION_FOLDER, 'upload')

DB_RUNTIME_PATH = os.path.join(WEB_EXECUTION_FOLDER, 'web_db')
DB_RUNTIME = SQLiteDb(DB_RUNTIME_PATH)

DB_IN_WORK = 'DB_IN_WORK'
app.config[DB_IN_WORK] = None


def submit_match(random_match):
    print(f'submitting {random_match}')
    client.submit(item=random_match[0],
                  frame=int(os.path.splitext(random_match[1].split('_')[1])[0]),
                  timecode=random_match[2])


def submit_random_match(matches: List[str]):
    if len(matches) == 0:
        print("Can't submit non-found item.")
    else:
        random_match = random.choice(matches)
        submit_match(random_match)


@app.route('/')
def index():
    return render_template('start_page.html')


@app.route('/use_stock_videos', methods=['POST'])
def use_stock_videos():
    """
    Setup DB_DEFAULT as DB_IN_WORK.

    :return: None
    """
    app.config[DB_IN_WORK] = DB_DEFAULT
    return redirect(url_for('render_main_page'))


@app.route('/upload', methods=['POST'])
def upload_videos():
    os.makedirs(WEB_EXECUTION_FOLDER, exist_ok=True)
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.config[DB_IN_WORK] = DB_RUNTIME
    app.config[DB_IN_WORK].create_db(force=True)

    files = request.files.getlist('file')
    if not files:
        return "No selected files"

    for file in files:
        if file.filename == '':
            continue

        file.save(os.path.join(UPLOAD_FOLDER, os.path.dirname(file.filename), file.filename))

    preprocess_videos(UPLOAD_FOLDER, NUM_KEY_FRAMES, DB_RUNTIME)

    return redirect(url_for('render_main_page'))


@app.route('/render_main_page')
def render_main_page():
    categories = GPRDataset().read_descriptions()
    categories = sorted(list(categories.values()))

    object_classes = sorted(YOLO_NAMES)

    return render_template('main.html', categories=categories, object_classes=object_classes)


@app.route('/handle_concept', methods=['POST'])
def handle_concept():
    selected_category = request.form.get('selected_category')

    print(f'target concept: {selected_category}')
    matches = app.config[DB_IN_WORK].get_rows_by_concept(selected_category)
    print(matches)
    submit_random_match(matches)

    return redirect(url_for('render_main_page'))


@app.route('/handle_object_class', methods=['POST'])
def handle_object_class():
    selected_object_class = request.form.get('selected_object_class')

    print(f'target object class: {selected_object_class}')
    matches = app.config[DB_IN_WORK].get_rows_by_object(selected_object_class)
    print(matches)
    submit_random_match(matches)

    return redirect(url_for('render_main_page'))


@app.route('/handle_random_object_class', methods=['POST'])
def handle_random_object_class():
    print(f'random object class')
    random_match = app.config[DB_IN_WORK].get_random_row_with_non_empty_objects()
    if random_match is not None:
        submit_match(random_match)
    else:
        print('There is no object in processed video. Nothing to submit')
    return redirect(url_for('render_main_page'))


@app.route('/handle_random_concept', methods=['POST'])
def handle_random_concept():
    print(f'random concept')
    random_match = app.config[DB_IN_WORK].get_random_row()
    if random_match is not None:
        submit_match(random_match)
    else:
        print('CRITICAL ERROR: your db is empty')
    return redirect(url_for('render_main_page'))


@app.route('/to_start_page')
def to_start_page():
    cleanup()
    return render_template('start_page.html')


@app.route('/cleanup', methods=['GET'])
def cleanup():
    if os.path.exists(WEB_EXECUTION_FOLDER):
        shutil.rmtree(WEB_EXECUTION_FOLDER)


@app.route('/exit', methods=['GET'])
def exit():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()

    cleanup()

    os._exit(0)


if __name__ == '__main__':
    app.run(debug=False)
