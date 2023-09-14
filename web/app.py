from flask import Flask, request, render_template, flash, redirect, url_for
import os
import threading

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

    return "Videos uploaded successfully."


if __name__ == '__main__':
    app.run(debug=False)
