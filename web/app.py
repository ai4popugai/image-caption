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

PROCESSING_THREAD = threading.Thread(target=preprocess_videos, args=(UPLOAD_FOLDER, 10, DB))


@app.route('/')
def index():
    return render_template('start_page.html')


@app.route('/upload', methods=['POST'])
def upload_videos():
    files = request.files.getlist('file')

    if not files:
        return "No selected files"

    for file in files:
        if file.filename == '':
            continue

        file.save(os.path.join(UPLOAD_FOLDER, os.path.dirname(file.filename), file.filename))

    PROCESSING_THREAD.start()

    # Redirect the user to the process_videos route
    return redirect(url_for('render_main_page'))


@app.route('/render_main_page')
def render_main_page():
    PROCESSING_THREAD.join()
    categories = GPRDataset().read_descriptions()
    return render_template('main.html',  categories=categories)


@app.route('/handle_category', methods=['POST'])
def handle_category():
    selected_category = request.form.get('selected_category')

    # Now you have the selected category, and you can perform any necessary actions with it.
    # For example, you can pass it to a function or store it in a session.

    # You can also redirect the user to another page based on their selection.
    # For now, let's just print the selected category to the console:
    print("Selected category:", selected_category)

    return redirect(url_for('render_main_page'))


if __name__ == '__main__':
    app.run(debug=False)
