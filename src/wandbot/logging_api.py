from fastapi import FastAPI, Request
from pydantic import BaseModel
from sqlite3 import connect
import wandb
from wandbot.config import TEAM, JOB_TYPE
from stream_table import StreamTable

app = FastAPI()
cols = ["source_name", "source_id", "wandb_run_id", "question", "response", "feedback", "elapsed_time", "start_time"]
wandb_stream_table = StreamTable('wandbot-results', cols)

class Response(BaseModel):
    source_name: str
    source_id: str
    wandb_run_id: str
    question: str
    response: str
    feedback: str
    elapsed_time: float
    start_time: float

def create_table():
    with connect("responses.db") as conn:
        cursor = conn.cursor()
        cursor.execute(
            """CREATE TABLE IF NOT EXISTS responses (
                        source_name TEXT,
                        source_id TEXT,
                        wandb_run_id TEXT,
                        question TEXT,
                        response TEXT,
                        feedback TEXT,
                        elapsed_time REAL,
                        start_time REAL
                      )"""
        )
        conn.commit()

create_table()

@app.post("/log")
async def log_response(response: Response):
    with connect("responses.db") as conn:
        source_name = response.source_name
        source_id = response.source_id
        wandb_run_id = response.wandb_run_id
        question = response.question
        response = response.response
        feedback = response.feedback
        elapsed_time = response.elapsed_time
        start_time = response.start_time
        try:
            wandb_stream_table.add_data(
                source_name, source_id, wandb_run_id, question, response, feedback, elapsed_time, start_time
            )
        except Exception as e:
            logger.error(e)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO responses (source_name, source_id, wandb_run_id, question, response, feedback, elapsed_time, start_time) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (response.source_name, response.source_id, response.wandb_run_id, response.question, response.response, response.feedback, response.elapsed_time, response.start_time),
        )
        conn.commit()
    return {"message": "Data logged successfully"}

@app.get("/log_db_to_wandb")
async def log_db_to_wandb():
    run = wandb.init(
            entity=TEAM,
            project="responses_table_dump",
            job_type=JOB_TYPE,
        )

    # Read data from the database
    with connect("collected_responses.db") as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM responses")
        data = cursor.fetchall()
    # Convert data to a wandb.Table
    columns = ["source_name", "source_id", "wandb_run_id", "question", "response", "feedback", "elapsed_time", "start_time"]
    wandb_table = wandb.Table(columns=columns, data=data)

    # Log the table to wandb
    run.log({"responses": wandb_table})

    # Create an artifact and add the table to it
    artifact = wandb.Artifact("responses", type="dataset")
    artifact.add(wandb_table, "responses_table")

    # Log the artifact
    run.log_artifact(artifact)

    # End the wandb run
    run.finish()

    return {"message": "Data logged to wandb successfully"}