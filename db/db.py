import glob
import os
import sqlite3

VIDEO_ID_KEY = 'video_id'
KEYFRAME_ID_KEY = 'keyframe_id'
OBJECTS_KEY = 'objects'
CONCEPT_KEY = 'concept'


class SQLiteDb:
    def __init__(self, db_name):
        self.db_dir = os.environ["SQLITE_DB_DIR"]

        self.version = len(glob.glob(os.path.join(self.db_dir, f"{db_name}*")))
        self.db_name = f"{db_name}_{self.version}"
        self.db_path = os.path.join(self.db_dir, self.db_name)
        self._create_db()

    def _create_db(self):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"CREATE TABLE {self.db_name}({VIDEO_ID_KEY},{KEYFRAME_ID_KEY},"
                           f"{OBJECTS_KEY},{CONCEPT_KEY})")
            connect.commit()

    def add_new_row(self, video_id, keyframe_id):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"INSERT INTO {self.db_name}({VIDEO_ID_KEY},{KEYFRAME_ID_KEY}) "
                           f"VALUES ('{video_id}', '{keyframe_id}')")
            connect.commit()

    def add_concept_to_row(self, video_id, keyframe_id, concept):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"UPDATE {self.db_name} SET {CONCEPT_KEY} = '{concept}' "
                           f"WHERE {VIDEO_ID_KEY} = '{video_id}' AND {KEYFRAME_ID_KEY} = '{keyframe_id}'")
            connect.commit()

    def add_objects_to_row(self, video_id, keyframe_id, objects):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"UPDATE {self.db_name} SET {OBJECTS_KEY} = '{objects}' "
                           f"WHERE {VIDEO_ID_KEY} = '{video_id}' AND {KEYFRAME_ID_KEY} = '{keyframe_id}'")
            connect.commit()
