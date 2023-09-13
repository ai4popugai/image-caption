import os


class SQLiteDb:
    def __init__(self, db_name):
        self.db_dir = os.environ["SQLITE_DB_DIR"]
        self.version = 0 if os.path.isfile(os.path.join(self.db_dir, db_name)) is False \
            else int(os.path.join(self.db_dir, db_name).split('_')[-1]) + 1
        self.db_name = f"{db_name}_{self.version}"
        self.db_path = os.path.join(self.db_dir, self.db_name)

