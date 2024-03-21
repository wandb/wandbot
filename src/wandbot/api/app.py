"""This module serves as the main server API for the wandbot application.

It imports and uses the FastAPI framework to define the API and initialize application event handlers like "startup".
Also, the module includes Python's built-in asyncio library for managing asynchronous tasks related to database backup.

The API includes:
- APICreateChatThreadRequest
- APIFeedbackRequest
- APIFeedbackResponse
- APIGetChatThreadResponse
- APIQueryRequest
- APIQueryResponse
- APIQuestionAnswerRequest
- APIQuestionAnswerResponse

Following classes and their functionalities:
- Chat: Main chat handling class, initialized during startup.
- ChatConfig: Configuration utility for chat.
- ChatRequest: Schema to handle requests made to the chat.

It also sets up and interacts with the database through:
- DatabaseClient: A utility to interact with the database.
- Base.metadata.create_all(bind=engine): Creates database tables based on the metadata.

The server runs periodic backup of the data to wandb using the backup_db method which runs as a coroutine.
The backup data is transformed into a Pandas DataFrame and saved as a wandb.Table.

It uses logger from the utils module for logging purposes.
"""

import os
os.environ["CONFIG_PATH"] = "config.yaml"

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import pandas as pd
import wandb
from fastapi import FastAPI
from wandbot.api.routers import chat as chat_router
from wandbot.api.routers import database as database_router
from wandbot.api.routers import retrieve as retrieve_router
from wandbot.ingestion.utils import get_embedding_config
from wandbot.retriever import VectorStore
from wandbot.utils import get_logger, load_config

logger = get_logger(__name__)
last_backup = datetime.now().astimezone(timezone.utc)

config = load_config(os.environ["CONFIG_PATH"])
logger.info(
    f"Loaded configuration from {os.environ['CONFIG_PATH']}: {config}"
)
vectorstore_config = get_embedding_config(
    config.embeddings.type, config.embeddings.config
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles the lifespan of the application.

    This function is called by the Uvicorn server to handle the lifespan of the application.
    It is used to perform any necessary startup and shutdown operations.

    Returns:
        None
    """
    vector_store = VectorStore.from_config(vectorstore_config)
    chat_router.chat = chat_router.Chat(vector_store=vector_store, config=config)
    database_router.db_client = database_router.DatabaseClient()
    if not config.retrieval_re_ranker.enabled:
        retrieve_router.retriever = retrieve_router.SimpleRetrievalEngine(
            vector_store=vector_store
        )
    else:
        retrieve_router.retriever = retrieve_router.SimpleRetrievalEngineWithRerank(
            vector_store=vector_store
        )


    async def backup_db():
        """Periodically backs up the database to a table.

        This function runs periodically and retrieves all question-answer threads from the database since the last backup.
        It then creates a pandas DataFrame from the retrieved threads and logs it to a table using Weights & Biases.
        The last backup timestamp is updated after each backup.

        Returns:
            None
        """
        global last_backup
        while True:
            chat_threads = database_router.db_client.get_all_question_answers(
                last_backup
            )
            if chat_threads is not None:
                chat_table = pd.DataFrame(
                    [chat_thread for chat_thread in chat_threads]
                )
                last_backup = datetime.now().astimezone(timezone.utc)
                logger.info(
                    f"Backing up database to Table at {last_backup}: Number of chat threads: {len(chat_table)}"
                )
                wandb.log(
                    {"question_answers_db": wandb.Table(dataframe=chat_table)}
                )
            await asyncio.sleep(600)

    _ = asyncio.create_task(backup_db())
    yield
    if wandb.run is not None:
        wandb.run.finish()


app = FastAPI(
    title="Wandbot", name="wandbot", version="1.3.0", lifespan=lifespan
)


app.include_router(chat_router.router)
app.include_router(database_router.router)
app.include_router(retrieve_router.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
