import wandb
import sqlite3

def read_db():
    conn = sqlite3.connect('responses.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM responses''')
    rows = cursor.fetchall()
    for row in rows:
        print(row)

read_db()

def convert_db_to_wandb_table():
    "Converts the SQLite database to a wandb.Table object"
    conn = sqlite3.connect('responses.db')
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM responses''')
    rows = cursor.fetchall()
    table = wandb.Table(columns=["discord_id", "wandb_run_id", "question", "response", "feedback"])
    for row in rows:
        table.add_data(*row)
    return table