import random
import shutil
from typing import List

from flask import Flask
import os

from db import SQLiteDb
from dres_api.client import Client

NUM_KEY_FRAMES = 25

app = Flask(__name__)
client = Client()


WEB_DEFAULTS_FOLDER = os.path.join(os.getcwd(), 'web_defaults')
DB_PATH = os.path.join(WEB_DEFAULTS_FOLDER, 'v3c1')
DB = SQLiteDb(DB_PATH)

