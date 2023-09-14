from flask import Flask, request, render_template, flash, redirect, url_for
import os
import threading

from web.web_functional import preprocess_videos

app = Flask(__name__)
app.secret_key = 'your_secret_key'

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'web', 'upload')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


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

        file.save(os.path.join(app.config['UPLOAD_FOLDER'], os.path.dirname(file.filename), file.filename))

    flash(f'{len(files)} video(s) uploaded successfully.')

    # Redirect the user to the process_videos route
    return redirect(url_for('process_uploaded_videos'))


@app.route('/process_videos')
def process_uploaded_videos():
    # Apply your preprocess_videos method to the UPLOAD folder
    preprocessing_thread = threading.Thread(target=preprocess_videos, args=(app.config['UPLOAD_FOLDER'], 50))
    preprocessing_thread.start()

    return "Video processing started."


if __name__ == '__main__':
    app.run(debug=False)
