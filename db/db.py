import glob
import os
import sqlite3

from db import VIDEO_ID_KEY, KEYFRAME_ID_KEY, OBJECTS_KEY, CONCEPT_KEY


class SQLiteDb:
    def __init__(self, db_name):
        self.db_dir = os.environ["SQLITE_DB_DIR"]

        self.version = len(glob.glob(os.path.join(self.db_dir, f"{db_name}*")))
        self.db_name = f"{db_name}_{self.version}"
        self._db_path = os.path.join(self.db_dir, self.db_name)
        self._create_db()

    def _create_db(self):
        with sqlite3.connect(self._db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"CREATE TABLE {self.db_name}({VIDEO_ID_KEY},{KEYFRAME_ID_KEY},"
                           f"{OBJECTS_KEY},{CONCEPT_KEY})")
