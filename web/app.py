from flask import Flask, request, render_template
import os

app = Flask(__name__)

# Adjust the UPLOAD_FOLDER path to reflect the new structure
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'web')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Update the path to the templates folder
app.template_folder = os.path.join(os.getcwd(), 'web', 'templates')


@app.route('/')
def index():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part"

    file = request.files['file']

    if file.filename == '':
        return "No selected file"

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        return "File uploaded successfully"


if __name__ == '__main__':
    app.run(debug=True)
