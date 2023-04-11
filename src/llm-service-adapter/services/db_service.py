# services/db_service.py
import sqlite3 #TODO: sqlite sucks replace with postgres
from wandb_utils.stream_table import StreamTable

class DBService:
    #TODO: parameterize the database name and table name
    def __init__(self, config: dict = {}):
        self.conn = sqlite3.connect("responses.db")
        self.cursor = self.conn.cursor()
        self.init_db()
        self.wandb_table = self.init_wandb_stream_table("wandbot-results")

    def init_db(self):
        self.cursor.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                    source_name TEXT,
                    source_id TEXT,
                    wandb_run_id TEXT,
                    query TEXT,
                    response TEXT,
                    feedback TEXT,
                    elapsed_time REAL,
                    start_time REAL
                  )"""
        )
        self.conn.commit()

    def init_wandb_stream_table(self, table_name):
        cols = [
            "source_name",
            "source_id",
            "wandb_run_id",
            "query",
            "response",
            "feedback",
            "elapsed_time",
            "start_time",
        ]
        return StreamTable(table_name, cols)

    def log_to_wandb_stream_table(self, source_name, source_id, run_id, query, response, feedback, elapsed_time, start_time):
        self.wandb_table.add_data(source_name, source_id, run_id, query, response, feedback, elapsed_time, start_time)

    def add_response(self, source_name, source_id, wandb_run_id, query, response, feedback, elapsed_time, start_time):
        self.cursor.execute(
            f"INSERT INTO responses (source_name, source_id, wandb_run_id, query, response, feedback, elapsed_time, start_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (source_name, source_id, wandb_run_id, query, response, feedback, elapsed_time, start_time),
        )
        self.conn.commit()
        self.log_to_wandb_stream_table(source_name, source_id, wandb_run_id, query, response, feedback, elapsed_time, start_time)

    def close(self):
        self.conn.close()
