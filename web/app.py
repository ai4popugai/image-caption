from flask import Flask, request, render_template, flash, redirect, url_for
import os
import threading

from datasets.classification.gpr import GPRDataset
from db import SQLiteDb
from web.web_functional import preprocess_videos

app = Flask(__name__)
app.secret_key = 'your_secret_key'

WEB_EXECUTION_FOLDER = os.path.join(os.getcwd(), 'web_exe')
os.makedirs(WEB_EXECUTION_FOLDER, exist_ok=True)

UPLOAD_FOLDER = os.path.join(WEB_EXECUTION_FOLDER, 'upload')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

DB_PATH = os.path.join(WEB_EXECUTION_FOLDER, 'db')
DB = SQLiteDb(DB_PATH)


@app.route('/')
def index():
    return render_template('main.html')


@app.route('/upload', methods=['POST'])
def upload_videos():
    files = request.files.getlist('file')

    if not files:
        return "No selected files"

    for file in files:
        if file.filename == '':
            continue

        file.save(os.path.join(UPLOAD_FOLDER, os.path.dirname(file.filename), file.filename))

    flash(f'{len(files)} video(s) uploaded successfully.')

    # Redirect the user to the process_videos route
    return redirect(url_for('process_uploaded_videos'))


@app.route('/process_videos')
def process_uploaded_videos():
    # Apply your preprocess_videos method to the UPLOAD folder
    preprocessing_thread = threading.Thread(target=preprocess_videos, args=(UPLOAD_FOLDER, 50, DB))
    preprocessing_thread.start()

    # Render the template with the dropdown menus
    return redirect(url_for('render_main_page'))


@app.route('/render_main_page')
def render_main_page():
    categories = GPRDataset().read_descriptions()
    print(categories)
    return render_template('categories.html',  categories=categories)


if __name__ == '__main__':
    app.run(debug=False)
