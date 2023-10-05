import glob
import os
import random
import shutil
import sqlite3

VIDEO_ID_KEY = 'video_id'
KEYFRAME_ID_KEY = 'keyframe_id'
TIMESTAMP_KEY = 'timestamp'
OBJECTS_KEY = 'objects'
CONCEPT_KEY = 'concept'
CAPTION_KEY = 'caption'


class SQLiteDb:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_name = os.path.basename(self.db_path)

    def create_db(self, force: bool = False):
        if os.path.isfile(self.db_path) is True:
            if force is True:
                os.remove(self.db_path)
            else:
                raise RuntimeError('Database already exists')
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"CREATE TABLE {self.db_name}({VIDEO_ID_KEY},{KEYFRAME_ID_KEY},{TIMESTAMP_KEY},"
                           f"{OBJECTS_KEY},{CONCEPT_KEY},{CAPTION_KEY})")
            connect.commit()

    def add_new_row(self, video_id, keyframe_id, timestamp):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"INSERT INTO {self.db_name}({VIDEO_ID_KEY},{KEYFRAME_ID_KEY},{TIMESTAMP_KEY}) "
                           f"VALUES ('{video_id}', '{keyframe_id}', '{timestamp}')")
            connect.commit()

    def add_caption_to_row(self, video_id, keyframe_id, caption):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"UPDATE {self.db_name} SET {CAPTION_KEY} = ? "
                           f"WHERE {VIDEO_ID_KEY} = '{video_id}' AND {KEYFRAME_ID_KEY} = '{keyframe_id}'",
                           (caption,))
            connect.commit()

    def add_concept_to_row(self, video_id, keyframe_id, concept):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"UPDATE {self.db_name} SET {CONCEPT_KEY} = ? "
                           f"WHERE {VIDEO_ID_KEY} = '{video_id}' AND {KEYFRAME_ID_KEY} = '{keyframe_id}'",
                           (concept,))
            connect.commit()

    def add_objects_to_row(self, video_id, keyframe_id, objects):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"UPDATE {self.db_name} SET {OBJECTS_KEY} = '{objects}' "
                           f"WHERE {VIDEO_ID_KEY} = '{video_id}' AND {KEYFRAME_ID_KEY} = '{keyframe_id}'")
            connect.commit()

    def get_rows_by_concept(self, target_concept):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"SELECT * FROM {self.db_name} WHERE {CONCEPT_KEY} = ?", (target_concept,))
            rows = cursor.fetchall()
            return rows

    def get_rows_by_object(self, target_object):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"SELECT * FROM {self.db_name} WHERE {OBJECTS_KEY} LIKE ?", (f'%"{target_object}":%',))
            rows = cursor.fetchall()
            return rows

    def get_random_row_with_non_empty_objects(self):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(
                f"SELECT * FROM {self.db_name} WHERE {OBJECTS_KEY} IS NOT NULL AND {OBJECTS_KEY} != '[]'")
            rows = cursor.fetchall()
            if rows:
                return random.choice(rows)
            else:
                return None

    def get_random_row(self):
        with sqlite3.connect(self.db_path) as connect:
            cursor = connect.cursor()
            cursor.execute(f"SELECT * FROM {self.db_name}")
            rows = cursor.fetchall()
            if rows:
                return random.choice(rows)
            else:
                return None