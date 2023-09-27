import atexit
import sys
import shutil

from flask import Flask, request, render_template, flash, redirect, url_for
import os
import threading

from datasets.classification.gpr import GPRDataset
from db import SQLiteDb
from nn_models.object_detection.yolov7.yolo_inference import YOLO_NAMES
from web.preprocess_video import preprocess_videos

NUM_KEY_FRAMES = 25

app = Flask(__name__)

WEB_DEFAULTS_FOLDER = os.path.join(os.getcwd(), 'web_defaults')
DB_DEFAULT_PATH = os.path.join(WEB_DEFAULTS_FOLDER, 'v3c1_unpacked')
DB_DEFAULT = SQLiteDb(DB_DEFAULT_PATH)

WEB_EXECUTION_FOLDER = os.path.join(os.getcwd(), 'web_runtime')
os.makedirs(WEB_EXECUTION_FOLDER, exist_ok=True)

UPLOAD_FOLDER = os.path.join(WEB_EXECUTION_FOLDER, 'upload')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_RUNTIME_PATH = os.path.join(WEB_EXECUTION_FOLDER, 'web_db')
DB_RUNTIME = SQLiteDb(DB_RUNTIME_PATH)

PROCESSING_THREAD = threading.Thread(target=preprocess_videos,
                                     args=(UPLOAD_FOLDER, NUM_KEY_FRAMES, DB_RUNTIME))
DB_IN_WORK = 'DB_IN_WORK'
app.config[DB_IN_WORK] = None


@app.route('/')
def index():
    return render_template('start_page.html')


@app.route('/upload', methods=['POST'])
def upload_videos():
    app.config[DB_IN_WORK] = DB_RUNTIME
    app.config[DB_IN_WORK].create_db()
    files = request.files.getlist('file')

    if not files:
        return "No selected files"

    for file in files:
        if file.filename == '':
            continue

        file.save(os.path.join(UPLOAD_FOLDER, os.path.dirname(file.filename), file.filename))

    PROCESSING_THREAD.start()
    PROCESSING_THREAD.join()

    # Redirect the user to the process_videos route
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

    return redirect(url_for('render_main_page'))


@app.route('/handle_object_class', methods=['POST'])
def handle_object_class():
    selected_object_class = request.form.get('selected_object_class')

    print(f'target object class: {selected_object_class}')
    matches = app.config[DB_IN_WORK].get_rows_by_object(selected_object_class)
    print(matches)

    return redirect(url_for('render_main_page'))


@app.route('/exit', methods=['GET'])
def cleanup():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is not None:
        func()

    # Delete the WEB_EXECUTION_FOLDER and its contents
    if os.path.exists(WEB_EXECUTION_FOLDER):
        shutil.rmtree(WEB_EXECUTION_FOLDER)

    # Exit the Python script
    os._exit(0)


if __name__ == '__main__':
    app.run(debug=False)
